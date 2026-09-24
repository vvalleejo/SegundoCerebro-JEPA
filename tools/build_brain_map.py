"""Genera el mapa interactivo del segundo cerebro (wiki/brain-map.html) y el lint de la bóveda.

Uso:
    python tools/build_brain_map.py            # genera wiki/brain-map.html
    python tools/build_brain_map.py --lint     # informe de salud por consola (no genera HTML)
    python tools/build_brain_map.py --lint --strict   # exit 1 si hay enlaces colgantes o LaTeX corrupto

Reglas de resolución de wikilinks (ver wiki/GEMINI.md §2):
    1. Destino con ruta ("wiki/papers/2026_X", "papers/2026_X"): se resuelve por sufijo de ruta.
    2. Nombre base único: se resuelve directamente.
    3. Nombre base duplicado (papers/ y architecture/): se marca como ambiguo y se resuelve a
       papers/, salvo que el origen sea esa misma nota de papers/ (entonces a architecture/).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("brain_map")

CONTENT_FOLDERS = ("papers", "architecture", "entities", "math", "synthesis")
META_NOTES = {"index", "log", "GEMINI", "repositories"}
PAPER_REQUIRED = (
    "title", "authors", "year", "venue", "arxiv", "source_pdf", "repo",
    "type", "family", "modality", "anti_collapse", "predictor", "planner", "tags",
)
ARCH_REQUIRED = ("title", "type", "paper", "tags")

_CODE_FENCE_RE = re.compile(r"^(```|~~~).*?^\1\s*$", re.M | re.S)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_WIKILINK_RE = re.compile(r"(!?)\[\[([^\]\n]+?)\]\]")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
# Restos de LaTeX cuyo "\t", "\f", "\a", "\r", "\b" se interpretó como escape al escribir el archivo.
_CORRUPT_LATEX_RE = re.compile(
    r"[\x07\x08\x0b\x0c]"
    r"|\t(?:heta|ext|au|imes|ilde|op|riangle|frac)\b"
    r"|(?<![\\A-Za-z])(?:rac\{|heta\b|lpha\b|pprox\b|ext\{|eta_|ho\b)"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "of", "for", "with", "and", "in", "to", "via", "from", "by", "on",
    "learning", "model", "models", "architecture", "architectures", "jepa", "joint", "embedding",
    "predictive", "pdf",
}


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict[str, Any] | None, str, str | None, bool]:
    """Separa el frontmatter YAML del cuerpo.

    Tolera BOM y líneas en blanco antes del primer ``---`` (y lo reporta como problema,
    porque Obsidian/Dataview no leen ese frontmatter).

    Args:
        text: Contenido completo de la nota.

    Returns:
        (frontmatter | None, cuerpo, error | None, leading_garbage) donde leading_garbage indica
        BOM o líneas en blanco antes del frontmatter.
    """
    leading_garbage = text.startswith("﻿")
    stripped = text.lstrip("﻿")
    candidate = stripped.lstrip("\r\n \t")
    if candidate != stripped and candidate.startswith("---"):
        leading_garbage = True
    if not candidate.startswith("---"):
        return None, stripped, None, False
    lines = candidate.splitlines(keepends=True)
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None, stripped, "frontmatter sin cierre '---'", leading_garbage
    raw = "".join(lines[1:end])
    body = "".join(lines[end + 1:])
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:  # frontmatter malformado: se reporta, no se aborta
        return None, body, f"YAML inválido: {exc.__class__.__name__}", leading_garbage
    if not isinstance(data, dict):
        return None, body, "el frontmatter no es un mapa", leading_garbage
    if "author" in data and "authors" not in data:
        data["authors"] = data.pop("author")
    return data, body, None, leading_garbage


def strip_code(text: str) -> str:
    """Elimina bloques de código e inline code (los enlaces de ejemplo no son aristas)."""
    return _INLINE_CODE_RE.sub("", _CODE_FENCE_RE.sub("", text))


def extract_wikilinks(text: str) -> list[tuple[str, bool]]:
    """Extrae destinos de wikilinks normalizados (sin ``|alias`` ni ``#heading``).

    Args:
        text: Texto markdown (ya sin código).

    Returns:
        Lista de (destino, es_embed).
    """
    out: list[tuple[str, bool]] = []
    for bang, inner in _WIKILINK_RE.findall(text):
        # En tablas Obsidian se escribe [[destino\|alias]]: se quita la barra de escape.
        target = inner.split("|", 1)[0].rstrip("\\").split("#", 1)[0].split("^", 1)[0].strip()
        target = target.replace("\\", "/")
        if target.lower().endswith(".md"):
            target = target[:-3]
        if target:
            out.append((target, bool(bang)))
    return out


def normalize_tag(tag: Any) -> str:
    """'#World_Models' -> 'worldmodels' (mismo criterio que el dataviewjs de la nota maestra)."""
    return re.sub(r"[-_\s]", "", str(tag).lower().lstrip("#"))


def title_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def token_similarity(a: str, b: str) -> float:
    """Proporción de tokens de ``a`` presentes en ``b`` (asimétrica: a = nombre corto del PDF)."""
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def first_paragraph(body: str, limit: int = 600) -> str:
    """Primer párrafo de prosa (salta encabezados, callouts vacíos, imágenes y tablas)."""
    text = strip_code(body)
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        lines = [ln[1:].strip() if ln.startswith(">") else ln for ln in lines]
        lines = [ln for ln in lines if ln and not ln.startswith(("#", "|", "![", "---", "[!"))]
        if not lines:
            continue
        para = " ".join(lines)
        para = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", para)
        para = re.sub(r"\*\*|__", "", para)
        if len(para) < 40:
            continue
        return para if len(para) <= limit else para[:limit].rsplit(" ", 1)[0] + "…"
    return ""


# --------------------------------------------------------------------------------------------
# Modelo de la bóveda
# --------------------------------------------------------------------------------------------

@dataclass
class Note:
    rel: str                      # "papers/2026_LpWM" (relativo a wiki/, sin .md)
    folder: str
    name: str
    fm: dict[str, Any] | None
    fm_error: str | None
    fm_leading_garbage: bool
    body: str
    raw_links: list[tuple[str, bool]] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


class LinkIndex:
    """Índice para resolver destinos de wikilinks a notas."""

    def __init__(self, notes: dict[str, Note]) -> None:
        self.notes = notes
        self.by_name: dict[str, list[str]] = {}
        for rel, note in notes.items():
            self.by_name.setdefault(note.name.lower(), []).append(rel)

    def resolve(self, target: str, source_rel: str) -> tuple[str | None, bool]:
        """Resuelve un destino.

        Returns:
            (rel | None si colgante, ambiguo)
        """
        t = target.strip().strip("/")
        if "/" in t:
            low = t.lower()
            if low.startswith("wiki/"):
                low = low[5:]
            matches = [r for r in self.notes if r.lower() == low or r.lower().endswith("/" + low)]
            return (matches[0] if len(matches) == 1 else None), len(matches) > 1
        cands = self.by_name.get(t.lower(), [])
        if not cands:
            return None, False
        if len(cands) == 1:
            return cands[0], False
        papers = [c for c in cands if c.startswith("papers/")]
        archs = [c for c in cands if c.startswith("architecture/")]
        if source_rel in papers and archs:
            return archs[0], True
        if papers:
            return papers[0], True
        return sorted(cands)[0], True


def load_notes(wiki: Path) -> dict[str, Note]:
    notes: dict[str, Note] = {}
    for path in sorted(wiki.rglob("*.md")):
        rel_path = path.relative_to(wiki)
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        rel = rel_path.with_suffix("").as_posix()
        folder = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("no UTF-8: %s", rel)
            text = path.read_text(encoding="utf-8", errors="replace")
        fm, body, err, garbage = parse_frontmatter(text)
        fm_text = text[: len(text) - len(body)] if fm is not None else ""
        clean = strip_code(fm_text + "\n" + body)
        note = Note(rel=rel, folder=folder, name=path.stem, fm=fm, fm_error=err,
                    fm_leading_garbage=garbage, body=body)
        note.raw_links = extract_wikilinks(clean)
        note.images = _MD_IMAGE_RE.findall(clean)
        notes[rel] = note
    return notes


def parse_repositories(wiki: Path) -> list[tuple[str, str]]:
    """Lee repositories.md -> [(título, url | '')]."""
    path = wiki / "repositories.md"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*-\s+\*\*(.+?)\*\*:\s*(.*)", line)
        if not m:
            continue
        url = re.search(r"\((https?://[^)\s]+)\)", m.group(2))
        out.append((m.group(1).strip(), url.group(1) if url else ""))
    return out


def as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def build_vault(vault: Path) -> dict[str, Any]:
    """Construye el modelo completo (nodos, aristas, salud) a partir de la bóveda.

    Args:
        vault: Raíz del vault (contiene wiki/ y raw/).

    Returns:
        Diccionario serializable a JSON.

    Raises:
        FileNotFoundError: Si no existe vault/wiki.
    """
    wiki = vault / "wiki"
    if not wiki.is_dir():
        raise FileNotFoundError(f"No existe {wiki}")
    notes = load_notes(wiki)
    index = LinkIndex(notes)

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    dangling: list[dict[str, str]] = []
    ambiguous: list[dict[str, str]] = []
    for rel, note in notes.items():
        for target, embed in note.raw_links:
            dest, amb = index.resolve(target, rel)
            if dest is None:
                dangling.append({"source": rel, "target": target})
                continue
            if amb:
                ambiguous.append({"source": rel, "target": target, "resolved": dest})
            if dest == rel:
                continue
            key = (rel, dest)
            edges.setdefault(key, {"source": rel, "target": dest, "count": 0})["count"] += 1

    incoming: dict[str, set[str]] = {r: set() for r in notes}
    outgoing: dict[str, set[str]] = {r: set() for r in notes}
    for (s, t) in edges:
        outgoing[s].add(t)
        incoming[t].add(s)

    repos = parse_repositories(wiki)
    raw_dir = vault / "raw"
    raw_pdfs = sorted(p.name for p in raw_dir.glob("*.pdf")) if raw_dir.is_dir() else []

    papers = {r: n for r, n in notes.items() if n.folder == "papers"}
    # PDF -> nota: primero source_pdf del frontmatter, luego similitud de tokens con el título.
    pdf_to_note: dict[str, str] = {}
    for rel, n in papers.items():
        src = (n.fm or {}).get("source_pdf") or ""
        if src:
            pdf_to_note[Path(str(src)).name] = rel
    for pdf in raw_pdfs:
        if pdf in pdf_to_note:
            continue
        scored = sorted(
            ((token_similarity(Path(pdf).stem, str((n.fm or {}).get("title", n.name))), r)
             for r, n in papers.items()),
            reverse=True,
        )
        if scored and scored[0][0] >= 0.5 and scored[0][1] not in pdf_to_note.values():
            pdf_to_note[pdf] = scored[0][1]
    note_to_pdf = {v: k for k, v in pdf_to_note.items()}

    def repo_for(rel: str) -> str:
        fm = notes[rel].fm or {}
        if fm.get("repo"):
            return str(fm["repo"])
        title = str(fm.get("title", notes[rel].name))
        best = max(repos, key=lambda rt: token_similarity(rt[0], title), default=None)
        if best and token_similarity(best[0], title) >= 0.6:
            return best[1]
        return ""

    img_dir = wiki / "architecture" / "img"
    all_imgs = sorted(p.name for p in img_dir.glob("*.png")) if img_dir.is_dir() else []
    used_imgs = {Path(i).name for n in notes.values() for i in n.images}

    master_rel = "synthesis/JEPA-master-note"
    in_master = outgoing.get(master_rel, set())

    nodes = []
    for rel, n in notes.items():
        fm = n.fm or {}
        is_meta = n.folder == "root" and n.name in META_NOTES
        year = fm.get("year")
        node = {
            "id": rel,
            "folder": n.folder,
            "name": n.name,
            "title": str(fm.get("title") or n.name),
            "meta": is_meta,
            "tags": sorted({normalize_tag(t) for t in as_list(fm.get("tags"))}),
            "year": int(year) if isinstance(year, int) or (isinstance(year, str) and year.isdigit()) else None,
            "authors": as_list(fm.get("authors")),
            "venue": str(fm.get("venue") or ""),
            "arxiv": str(fm.get("arxiv") or ""),
            "family": str(fm.get("family") or ""),
            "modality": as_list(fm.get("modality")),
            "anti_collapse": as_list(fm.get("anti_collapse")),
            "predictor": fm.get("predictor"),
            "planner": as_list(fm.get("planner")),
            "provenance": str(fm.get("provenance") or ""),
            "status": str(fm.get("status") or ""),
            "excerpt": first_paragraph(n.body),
            "images": [f"architecture/{i}" if not i.startswith("architecture/") else i
                       for i in n.images] if n.folder == "architecture" else [],
            "pdf": note_to_pdf.get(rel, ""),
            "repo": repo_for(rel) if n.folder == "papers" else "",
            "in": sorted(incoming[rel]),
            "out": sorted(outgoing[rel]),
            "words": len(n.body.split()),
        }
        if n.folder == "math":
            eq = _DISPLAY_MATH_RE.search(strip_code(n.body))
            node["equation"] = eq.group(1).strip() if eq else ""
        nodes.append(node)

    health = lint_vault(notes, incoming, dangling, ambiguous, raw_pdfs, pdf_to_note,
                        all_imgs, used_imgs, papers, in_master)
    return {
        "generated": date.today().isoformat(),
        "nodes": nodes,
        "edges": list(edges.values()),
        "health": health,
        "raw_pdfs": [{"pdf": p, "note": pdf_to_note.get(p, "")} for p in raw_pdfs],
    }


def lint_vault(
    notes: dict[str, Note],
    incoming: dict[str, set[str]],
    dangling: list[dict[str, str]],
    ambiguous: list[dict[str, str]],
    raw_pdfs: list[str],
    pdf_to_note: dict[str, str],
    all_imgs: list[str],
    used_imgs: set[str],
    papers: dict[str, Note],
    in_master: set[str],
) -> dict[str, Any]:
    """Calcula el informe de salud (ver GEMINI.md §7)."""
    content = {r: n for r, n in notes.items() if n.folder in CONTENT_FOLDERS}
    orphans = sorted(r for r in content if not incoming[r])
    only_index = sorted(r for r in content if incoming[r] == {"index"})

    fm_issues: list[dict[str, str]] = []
    for rel, n in notes.items():
        if n.folder == "root" and n.name in {"log", "repositories"}:
            continue
        if n.fm_error:
            fm_issues.append({"note": rel, "issue": n.fm_error})
        elif n.fm is None:
            fm_issues.append({"note": rel, "issue": "sin frontmatter"})
        if n.fm_leading_garbage:
            fm_issues.append({"note": rel, "issue": "BOM o líneas en blanco antes de '---'"})
        required = PAPER_REQUIRED if n.folder == "papers" else ARCH_REQUIRED if n.folder == "architecture" else ()
        missing = [k for k in required if n.fm is not None and k not in n.fm]
        if missing:
            fm_issues.append({"note": rel, "issue": "faltan: " + ", ".join(missing)})

    corrupt: list[dict[str, str]] = []
    for rel, n in notes.items():
        for m in _CORRUPT_LATEX_RE.finditer(strip_code(n.body)):
            start = max(0, m.start() - 25)
            snippet = n.body and strip_code(n.body)[start:m.end() + 25]
            corrupt.append({"note": rel, "snippet": snippet.replace("\n", " ").encode("unicode_escape").decode()})
            break

    stale = sorted(r for r in papers if r not in in_master)
    return {
        "dangling": dangling,
        "ambiguous": ambiguous,
        "orphans": orphans,
        "only_index": only_index,
        "frontmatter": fm_issues,
        "uningested_pdfs": [p for p in raw_pdfs if p not in pdf_to_note],
        "unused_images": [i for i in all_imgs if i not in used_imgs],
        "corrupt_latex": corrupt,
        "stale_synthesis": stale,
    }


def format_lint(health: dict[str, Any]) -> str:
    """Informe legible para consola."""
    amb_by_source: dict[str, int] = {}
    for d in health["ambiguous"]:
        amb_by_source[d["source"]] = amb_by_source.get(d["source"], 0) + 1
    sections = [
        ("Enlaces colgantes", [f"{d['source']} -> [[{d['target']}]]" for d in health["dangling"]]),
        (f"Enlaces ambiguos sin ruta ({len(health['ambiguous'])} enlaces; deuda técnica, GEMINI §2). "
         "Notas de origen",
         [f"{s}: {c}" for s, c in sorted(amb_by_source.items(), key=lambda kv: -kv[1])[:15]]
         + ([f"… y {len(amb_by_source) - 15} notas más"] if len(amb_by_source) > 15 else [])),
        ("Huérfanas (sin enlaces entrantes)", health["orphans"]),
        ("Solo enlazadas desde index", health["only_index"]),
        ("Frontmatter", [f"{f['note']}: {f['issue']}" for f in health["frontmatter"]]),
        ("PDFs de raw/ sin nota", health["uningested_pdfs"]),
        ("Imágenes sin usar", health["unused_images"]),
        ("LaTeX corrupto", [f"{c['note']}: …{c['snippet']}…" for c in health["corrupt_latex"]]),
        ("Papers ausentes de la nota maestra", health["stale_synthesis"]),
    ]
    out = []
    for title, items in sections:
        out.append(f"\n## {title}: {len(items)}")
        out.extend(f"  - {i}" for i in items[:200])
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# Render HTML
# --------------------------------------------------------------------------------------------

def inline_katex_css(vendor: Path) -> str:
    """Incrusta las fuentes woff2 de KaTeX como data URIs para que funcione offline."""
    css = (vendor / "katex.min.css").read_text(encoding="utf-8")

    def repl(m: re.Match[str]) -> str:
        font = vendor / m.group(1)
        if not font.exists():
            return "url(data:,)"
        b64 = base64.b64encode(font.read_bytes()).decode("ascii")
        return f"url(data:font/woff2;base64,{b64})"

    css = re.sub(r"url\((fonts/[^)]+\.woff2)\)", repl, css)
    # Las variantes woff/ttf no se distribuyen: se eliminan para no provocar peticiones.
    return re.sub(r",\s*url\(fonts/[^)]+\.(?:woff|ttf)\)\s*format\(\"(?:woff|truetype)\"\)", "", css)


def render_html(data: dict[str, Any], vendor: Path, template: Path) -> str:
    """Inserta datos y librerías en la plantilla HTML."""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = template.read_text(encoding="utf-8")
    replacements = {
        "/*__KATEX_CSS__*/": inline_katex_css(vendor),
        "/*__D3_JS__*/": (vendor / "d3.min.js").read_text(encoding="utf-8"),
        "/*__KATEX_JS__*/": (vendor / "katex.min.js").read_text(encoding="utf-8"),
        "/*__DATA__*/": payload,
    }
    for key, value in replacements.items():
        if key not in html:
            raise ValueError(f"Plantilla sin marcador {key}")
        html = html.replace(key, value.replace("</script", "<\\/script"))
    return html


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vault", type=Path, default=here.parent, help="raíz del vault")
    parser.add_argument("--out", type=Path, default=None, help="HTML de salida (def. wiki/brain-map.html)")
    parser.add_argument("--lint", action="store_true", help="solo informe de salud")
    parser.add_argument("--strict", action="store_true", help="exit 1 si hay colgantes o LaTeX corrupto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = build_vault(args.vault)
    health = data["health"]
    if args.lint:
        print(format_lint(health))
    else:
        out = args.out or args.vault / "wiki" / "brain-map.html"
        html = render_html(data, here / "vendor", here / "brain_map_template.html")
        out.write_text(html, encoding="utf-8")
        logger.info("Generado %s (%d nodos, %d aristas, %.0f KB)", out, len(data["nodes"]),
                    len(data["edges"]), len(html.encode("utf-8")) / 1024)
    if args.strict and (health["dangling"] or health["corrupt_latex"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

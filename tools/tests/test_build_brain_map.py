"""Tests del generador del mapa y del lint de la bóveda (pytest)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_brain_map as bbm  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """Bóveda mínima con nombres duplicados papers/architecture y casos límite."""
    _write(tmp_path, "raw/My Paper Title Long.pdf", "%PDF")
    _write(tmp_path, "raw/Orphan Unmatched Document.pdf", "%PDF")
    _write(tmp_path, "wiki/papers/2026_X.md",
           "---\ntitle: \"My Paper Title Long\"\nauthor: [A B]\nyear: 2026\ntags: [paper]\n---\n\n"
           "Resumen con un enlace a [[SIGReg]] y a la [[2026_X|arquitectura]].\n")
    _write(tmp_path, "wiki/architecture/2026_X.md",
           "---\ntitle: Arq\ntype: architecture\npaper: \"[[wiki/papers/2026_X|2026_X]]\"\ntags: [architecture]\n---\n"
           "![fig](img/2026_X_arch.png)\nVer [[2026_X]] y [[Inexistente]].\n")
    _write(tmp_path, "wiki/architecture/img/2026_X_arch.png", "png")
    _write(tmp_path, "wiki/architecture/img/unused.png", "png")
    _write(tmp_path, "wiki/math/SIGReg.md",
           "\n---\ntitle: SIGReg\ntags: [math]\n---\n\n$$T=B\\int|\\hat\\varphi-\\varphi|^2w$$\n"
           "Tabla: [[wiki/papers/2026_X\\|X]]. Código `[[NoEsEnlace]]`.\n\n```\n[[TampocoEsEnlace]]\n```\n")
    _write(tmp_path, "wiki/entities/Bad.md", "---\ntitle: Bad\ntags: [entity]\n---\nCorrupto: $\text{sg}$ y \x07pprox.\n")
    _write(tmp_path, "wiki/synthesis/JEPA-master-note.md", "---\ntitle: M\ntags: [synthesis]\n---\n[[SIGReg]]\n")
    _write(tmp_path, "wiki/index.md", "---\ntitle: Index\n---\n[[SIGReg]] [[Bad]] [[JEPA-master-note]]\n")
    _write(tmp_path, "wiki/repositories.md", "- **My Paper Title Long**: [https://github.com/x/y](https://github.com/x/y)\n")
    return tmp_path


# ---------------- parsing ----------------

def test_frontmatter_tolerates_bom_and_blank_line() -> None:
    fm, body, err, garbage = bbm.parse_frontmatter("﻿\n---\ntitle: T\n---\nbody")
    assert fm == {"title": "T"} and err is None and garbage is True and body.strip() == "body"


def test_frontmatter_normalizes_author_key() -> None:
    fm, *_ = bbm.parse_frontmatter("---\nauthor: [A]\n---\n")
    assert fm == {"authors": ["A"]}


def test_frontmatter_invalid_yaml_is_reported_not_raised() -> None:
    fm, _, err, _ = bbm.parse_frontmatter("---\ntitle: [unclosed\n---\n")
    assert fm is None and err and "YAML" in err


def test_extract_wikilinks_strips_alias_heading_and_table_escape() -> None:
    links = bbm.extract_wikilinks("[[A|x]] [[B#sec]] ![[C.md]] [[wiki/papers/D\\|d]]")
    assert links == [("A", False), ("B", False), ("C", True), ("wiki/papers/D", False)]


def test_strip_code_removes_example_links() -> None:
    text = bbm.strip_code("x `[[No]]` y\n```\n[[TampocoNo]]\n```\n[[Si]]")
    assert [t for t, _ in bbm.extract_wikilinks(text)] == ["Si"]


# ---------------- resolución ----------------

def test_resolution_rules(vault: Path) -> None:
    idx = bbm.LinkIndex(bbm.load_notes(vault / "wiki"))
    assert idx.resolve("wiki/papers/2026_X", "math/SIGReg") == ("papers/2026_X", False)
    assert idx.resolve("2026_X", "architecture/2026_X") == ("papers/2026_X", True)
    assert idx.resolve("2026_X", "papers/2026_X") == ("architecture/2026_X", True)
    assert idx.resolve("SIGReg", "papers/2026_X") == ("math/SIGReg", False)
    assert idx.resolve("Inexistente", "papers/2026_X") == (None, False)


# ---------------- modelo + lint ----------------

def test_build_vault_health(vault: Path) -> None:
    data = bbm.build_vault(vault)
    h = data["health"]
    assert {"source": "architecture/2026_X", "target": "Inexistente"} in h["dangling"]
    assert not any(d["target"] in {"NoEsEnlace", "TampocoEsEnlace"} for d in h["dangling"])
    assert h["uningested_pdfs"] == ["Orphan Unmatched Document.pdf"]
    assert h["unused_images"] == ["unused.png"]
    assert [c["note"] for c in h["corrupt_latex"]] == ["entities/Bad"]
    assert any(f["note"] == "math/SIGReg" and "BOM" in f["issue"] for f in h["frontmatter"])
    assert "papers/2026_X" in h["stale_synthesis"]


def test_build_vault_nodes(vault: Path) -> None:
    nodes = {n["id"]: n for n in bbm.build_vault(vault)["nodes"]}
    paper = nodes["papers/2026_X"]
    assert paper["authors"] == ["A B"]
    assert paper["pdf"] == "My Paper Title Long.pdf"
    assert paper["repo"] == "https://github.com/x/y"
    assert "architecture/2026_X" in paper["in"]
    assert nodes["architecture/2026_X"]["images"] == ["architecture/img/2026_X_arch.png"]
    assert nodes["math/SIGReg"]["equation"].startswith("T=B")
    assert nodes["index"]["meta"] is True


def test_corrupt_latex_regex_has_no_false_positive_on_clean_latex() -> None:
    clean = r"$\theta \frac{a}{b} \alpha \approx \text{x} \beta_1 \rho$"
    assert not bbm._CORRUPT_LATEX_RE.search(clean)


# ---------------- render ----------------

def test_render_html_embeds_data_and_escapes_script(vault: Path, tmp_path: Path) -> None:
    data = bbm.build_vault(vault)
    data["nodes"][0]["excerpt"] = "</script><script>alert(1)</script>"
    html = bbm.render_html(data, TOOLS / "vendor", TOOLS / "brain_map_template.html")
    payload = re.search(r'<script id="brain-data" type="application/json">(.*?)</script>', html, re.S)
    assert payload, "falta el bloque de datos"
    parsed = json.loads(payload.group(1).replace("<\\/", "</"))
    assert len(parsed["nodes"]) == len(data["nodes"])
    assert "</script><script>alert" not in html
    assert "/*__" not in html  # todos los marcadores sustituidos
    assert "data:font/woff2;base64," in html  # KaTeX offline


def test_main_lint_strict_exit_code(vault: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert bbm.main(["--vault", str(vault), "--lint", "--strict"]) == 1
    assert "Enlaces colgantes" in capsys.readouterr().out

---
title: "Esquema de la Wiki: World Models Second Brain"
type: schema
tags: [schema, meta]
---

# LLM Wiki Schema: Yann LeCun World Models Brain

Este archivo define las reglas, la estructura y el flujo de trabajo para mantener el "World Models Second Brain". El usuario es Ingeniero Matemático y está haciendo un doctorado en World Models, centrado en las arquitecturas JEPA de Yann LeCun y su ecosistema. **El objetivo es el rigor verificable:** cada afirmación técnica debe poder rastrearse hasta su fuente.

---

## 1. Estructura de la bóveda

La **raíz del vault de Obsidian** es la carpeta `Cerebro Doctorado/` (ahí vive `.obsidian/`). Toda ruta de wikilink o de Dataview se escribe relativa a esa raíz (`wiki/papers/...`).

| Ruta | Contenido | Convención de nombre |
|---|---|---|
| `raw/` | PDFs fuente **inmutables**. No se modifican ni se renombran. | Nombre original del PDF |
| `wiki/papers/` | Nota-resumen de cada paper | `YYYY_Short_Title.md` |
| `wiki/architecture/` | Desglose técnico del pipeline (encoders, predictor, masking, condicionamiento, loss, inferencia/planificación) | `YYYY_Short_Title.md` (existentes) · `YYYY_Short_Title_Arch.md` (**nuevas**) |
| `wiki/architecture/img/` | Figuras extraídas a 300 DPI | `YYYY_Short_Title_<vista>.png` |
| `wiki/math/` | Una página por formulación (loss, regularizador, teorema, cota) | `Concept_Name.md` |
| `wiki/entities/` | Arquitecturas/modelos con nombre propio (hub por modelo) y baselines externos | `Nombre.md` |
| `wiki/synthesis/` | Visiones transversales. Nota maestra: `JEPA-master-note.md` | libre |
| `wiki/index.md` | Map of Content (MOC). Se actualiza con **cada** archivo nuevo | — |
| `wiki/log.md` | Registro cronológico **append-only** | — |
| `wiki/repositories.md` | Código oficial / página de proyecto por paper | — |
| `wiki/brain-map.html` | Mapa interactivo **generado** (no editar a mano) | — |
| `tools/` | Scripts de mantenimiento (`build_brain_map.py`) y sus tests | — |

---

## 2. Nombres y enlaces (desambiguación obligatoria)

Existen 24 nombres base duplicados entre `papers/` y `architecture/` (p. ej. `2026_LpWM.md` en ambas). Un `[[2026_LpWM]]` sin ruta es **ambiguo** y Obsidian lo resuelve de forma no controlada.

- **Enlaces entre carpetas a notas con nombre duplicado:** usar siempre la ruta completa con alias:
  `[[wiki/papers/2026_LpWM|2026_LpWM]]`, `[[wiki/architecture/2026_LpWM|arquitectura LpWM]]`.
- **Enlaces a nombres únicos** (`entities/`, `math/`, `synthesis/`): basta `[[SIGReg]]`.
- **Notas nuevas:** su nombre base no puede coincidir con ningún archivo existente. En `architecture/` se usa el sufijo `_Arch`.
- **Frontmatter de `architecture/`:** `paper:` y `entity:` son la fuente de verdad de la arista arquitectura → paper/entidad (con ruta completa).
- **Deuda técnica registrada:** los 24 pares heredados no se renombran para no romper historial ni generar un diff masivo. Si algún día se hace, debe ser desde Obsidian (que reescribe los enlaces) y con entrada `refactor` en `log.md`.
- **Prohibido** enlazar por el título del PDF (`[[Giving Sensors a Voice]]`): se enlaza la nota (`[[CHARM]]`, `[[wiki/papers/2026_CHARM|2026_CHARM]]`).
- Un modelo externo que se cita a menudo como baseline (DINO-WM, Ctrl-World…) recibe un **stub** en `entities/` con `status: external-baseline`, en vez de quedar como enlace colgante.

---

## 3. Frontmatter (esquema por carpeta)

Reglas generales:
- UTF-8 **sin BOM**. El archivo empieza **exactamente** por `---` en la línea 1, sin línea en blanco previa, porque si no Dataview no lee el frontmatter.
- `tags` en **kebab-case y minúsculas** (`world-models`, `jepa`, `sigreg`). El primer tag es el tipo de nota.
- Las listas van como listas YAML, no como texto separado por comas.

### `papers/`
```yaml
---
title: "Título exacto del paper"
authors: [Nombre Apellido, ...]      # siempre 'authors', nunca 'author'
year: 2026
venue: "arXiv preprint"              # o "NeurIPS 2025", etc.
arxiv: "2609.10464"                  # "" si el PDF no lo indica
source_pdf: "raw/<nombre exacto>.pdf"
repo: "https://github.com/..."       # "" si no hay código
type: paper
family: jepa                         # jepa | jepa-adjacent | generative | review
modality: [video, control]           # image | video | audio | text | time-series | graph | control | theory | multimodal
anti_collapse: [sigreg]              # ema-sg | sg (stop-grad sin EMA) | vicreg | sigreg | rdmreg | disreg | kl | frozen-encoder | none | n/a
predictor: true                      # ¿hay red predictora explícita?
planner: [cem]                       # cem | mppi | gd | diffusion-policy | inverse | tta | vocab-search | none
tags: [paper, jepa, world-models]
---
```

### `architecture/`
```yaml
---
title: "Arquitectura X"
type: architecture
paper: "[[wiki/papers/2026_X|2026_X]]"
entity: "[[X]]"
tags: [architecture, jepa]
---
```

### `math/`
```yaml
---
title: "SIGReg"
type: math
used_by: ["[[LeJEPA]]", "[[LeWorldModel]]"]
tags: [math, regularizer]
---
```

### `entities/`, `synthesis/`
`title`, `type` (`entity` | `synthesis`), `tags`. Los stubs externos añaden `status: external-baseline`.

---

## 4. Reglas de ingesta

Cuando el usuario pide ingerir un PDF de `raw/`, se siguen estos pasos:

1. **Control de procedencia (antes de nada).** Registrar autoría, afiliaciones, venue, arXiv id y tipo de trabajo (`empírico` | `teórico` | `revisión` | `conceptual` | `librería`). Señalar las **señales de alerta** en un callout `> [!warning] Procedencia`: autoría no verificable o atípica (p. ej. un autor célebre como autor único de un trabajo fuera de su línea), ausencia de experimentos, ausencia de venue/arXiv, afirmaciones sin respaldo. La nota se crea igualmente, pero su `family` y su peso en las síntesis reflejan esa evaluación.
2. **Leer el PDF.** Identificar las contribuciones, la arquitectura, la matemática y los resultados. Leer las secciones concretas, no solo el abstract.
3. **Nota de paper** en `papers/` con el frontmatter completo (§3). Secciones: Resumen ejecutivo · Arquitectura · Formulación · Resultados clave · Limitaciones (declaradas por los autores y observadas) · Referencias cruzadas.
4. **Arquitectura y diagramas.** Extraer las figuras de arquitectura/pipeline a **300 DPI PNG** en `architecture/img/`. Crear la nota de arquitectura con las imágenes embebidas (`![descripción](img/archivo.png)`) y el detalle de encoders, predictor, masking, condicionamiento, losses e inferencia/planificación.
5. **Matemáticas.** Crear o actualizar páginas en `math/` con LaTeX estricto (`$...$` / `$$...$$`). **Definir cada variable** (tipo y dimensión). Si la formulación ya existe en `math/` (p. ej. SIGReg), **no se duplica**: se enlaza y se anota solo la diferencia.
6. **Entidades.** Si el paper introduce una arquitectura con nombre propio, crear o actualizar su página en `entities/`.
7. **Verificación (obligatoria).** Contrastar contra el PDF **cada ecuación y cada cifra** escrita en las notas nuevas:
   - Cada cifra lleva su fuente entre paréntesis: `(PDF §4.2)`, `(Eq. 7)`, `(Tab. 3)`, `(Fig. 2)`.
   - Lo que no se pueda verificar se marca con `> [!question] No verificado` o se elimina. **Nunca se infiere del título.**
   - Revisar si hay **contradicciones con notas existentes** (§6) y resolverlas o registrarlas.
8. **Referencias cruzadas** con wikilinks siguiendo §2: paper ↔ arquitectura ↔ matemáticas ↔ entidad ↔ baselines.
9. **Cierre de la ingesta:** actualizar `index.md`, `repositories.md` (enlace oficial o "Código no liberado") y la nota maestra `synthesis/JEPA-master-note.md` (tabla de §8 y secciones afectadas). Añadir la entrada en `log.md` y regenerar el mapa: `python tools/build_brain_map.py`.
10. **Log:** `## [YYYY-MM-DD] ingest | <Nombre>` con **Resumen**, **Acciones** y la lista de archivos creados o actualizados.

---

## 5. Formato, notación y estilo

- **Idioma:** narrativa en **español**; los términos técnicos se mantienen en **inglés** (Joint-Embedding Predictive Architecture, Energy-Based Model, stop-gradient, rollout...).
- **Rigor matemático alto.** No simplificar: desglosar losses, distribuciones latentes, hipótesis de los teoremas y cotas. Un teorema se enuncia **con sus hipótesis**.
- **Notación canónica de la bóveda** (usada en `synthesis/`; las notas de paper pueden conservar la notación del paper pero deben declarar la equivalencia):

| Símbolo | Significado |
|---|---|
| $B$ | tamaño de batch (muestras sobre las que se estima una distribución) |
| $N$ | nº de tokens (parches, pasos, nodos) |
| $D$ | dimensión del embedding |
| $K$ | nº de direcciones del sketch (slices) en SIGReg/RDMReg |
| $T$ | nº de nodos de cuadratura |
| $V_g, V_l, V$ | vistas globales, locales y totales |
| $H$ | horizonte de planificación |
| $\lambda$ | peso del regularizador. **Indicar siempre** si es aditivo ($\mathcal L_{\text{pred}}+\lambda\mathcal R$) o convexo ($(1-\lambda)\mathcal L_{\text{pred}}+\lambda\mathcal R$), porque sus valores no son comparables entre sí |

- **Normas:** escribir $\|\cdot\|_2^2$ cuando es MSE; no decir "L2" a secas si la fórmula está al cuadrado.
- **Higiene de LaTeX:** al generar archivos desde código, escribir el texto como *raw string* o escapar las barras. `\t`, `\f`, `\r`, `\a`, `\b` convertidos en caracteres de control corrompen `\theta`, `\frac`, `\rho`, `\alpha` y `\beta` (esto ya ocurrió en MotionJEPA). El lint lo detecta.
- **Contradicciones y evolución:** si un paper descarta una técnica anterior (p. ej. EMA/stop-gradient → SIGReg; contrastivo → regularizado), se anota explícitamente y se actualizan las páginas afectadas.
- **Callouts estándar:** `[!abstract]` TL;DR · `[!warning]` limitación o procedencia · `[!question]` no verificado · `[!important]` detalle que suele malinterpretarse.

---

## 6. Resolución de contradicciones

Cuando dos notas discrepan:
1. **Manda el PDF**, con cita de sección, ecuación o tabla.
2. Se corrige la nota errónea con una edición quirúrgica. No se reescribe entera.
3. La decisión se registra en la tabla *Decisiones de arbitraje* de `synthesis/JEPA-master-note.md` y en `log.md` (tipo `lint`).
4. Las definiciones canónicas viven en `math/` (p. ej. SIGReg = test de Epps–Pulley sobre proyecciones aleatorias). Cualquier otra nota que las describa debe ser coherente con ellas o enlazarlas.

---

## 7. Mantenimiento (lint)

Ejecutar `python tools/build_brain_map.py --lint` periódicamente y tras cada ingesta. Comprueba:
- Enlaces colgantes (destinos inexistentes) y enlaces ambiguos sin ruta a nombres duplicados.
- Páginas huérfanas (sin enlaces entrantes).
- Frontmatter ausente, inválido o incompleto según §3.
- PDFs de `raw/` sin nota en `papers/`.
- Imágenes de `architecture/img/` no embebidas.
- LaTeX corrupto: caracteres de control (TAB, BEL, BS, FF) seguidos de letras, o restos sin barra como `rac{`, `lpha`, `pprox`.
- Síntesis desactualizadas: papers que no aparecen en `synthesis/JEPA-master-note.md`.
- Variables matemáticas sin definir (revisión manual).

Tras el lint o una corrección: entrada en `log.md` `## [YYYY-MM-DD] lint | <alcance>`.

---

## 8. Mapa interactivo

`wiki/brain-map.html` se genera con `python tools/build_brain_map.py` a partir del frontmatter, los wikilinks, las imágenes, `raw/` y `repositories.md`. Es autocontenido y funciona offline. **Regenerarlo** tras cada ingesta, lint o corrección. Si una vista del mapa sale vacía o incorrecta, casi siempre la causa es un frontmatter incompleto (§3).

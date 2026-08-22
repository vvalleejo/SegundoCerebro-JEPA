# LLM Wiki Schema: Yann LeCun World Models Brain

This file defines the rules, structure, and workflow for maintaining the "World Models Second Brain" wiki. The user is a Mathematical Engineer pursuing a PhD in World Models (specifically focusing on Yann LeCun's JEPA architectures).

## Estructura de la Wiki (Wiki Structure)
- `raw/`: Immutable source PDFs. Do NOT modify.
- `papers/`: Summaries and detailed notes of each paper read. Name format: `[Year]_[Short_Title].md`.
- `architecture/`: Architectural diagrams, pipeline schematics, and deep technical breakdowns of model workflows and inference mechanisms. Diagram images are stored in `architecture/img/`. Name format: `[Year]_[Short_Title].md`.
- `math/`: Isolated concept pages for mathematical formulations, loss functions, optimizations, and proofs. Name format: `[Concept_Name].md`.
- `entities/`: Pages for specific architectures (e.g., JEPA, V-JEPA) or authors.
- `synthesis/`: High-level overviews connecting different concepts and tracking the evolution of ideas.
- `index.md`: The Map of Content (MOC). Must be updated whenever a new file is added.
- `log.md`: Chronological log of actions. Append-only.

## Reglas de Ingesta (Ingestion Rules)
When the user asks to ingest a new paper from `raw/`, follow these steps:
1. **Read the PDF**: Extract text and identify the core contributions, architecture, and math.
2. **Create Paper Page**: In `papers/`, create a summary. Include YAML frontmatter with `title`, `authors`, `year`, `tags`.
3. **Extract Architecture & Diagrams**: Extract the architectural diagrams and pipeline figures directly from the PDF in high resolution (300 DPI PNG) into `architecture/img/`. Create the corresponding note in `architecture/[Year]_[Short_Title].md` with embedded images, explaining in detail the encoders, predictor, masking strategy, conditioning, loss functions, and inference/planning mechanisms.
4. **Extract Math**: Identify key equations (e.g., Energy functions, predictive losses, contrastive losses). Create or update pages in `math/`. Use strict LaTeX `$equation$` or `$$equation$$` blocks. Explain the variables rigorously.
5. **Update Entities**: If the paper introduces a new architecture (e.g., I-JEPA), create/update its page in `entities/`.
6. **Cross-reference**: Extensively use Obsidian-style wiki links `[[Page Name]]`. Connect the paper to its architecture note, math concepts, and entities.
7. **Update Index**: Add links to the newly created files in `index.md`.
8. **Log**: Append an entry to `log.md` with the format: `## [YYYY-MM-DD] ingest | [Paper Name]`

## Formato y Estilo (Format & Style)
- **Idioma**: Narrativa y explicaciones en **Español** (Spanish), pero mantén los términos técnicos comunes en **Inglés** (e.g., "Joint-Embedding Predictive Architecture", "Energy-Based Model", "Latent Masking").
- **Rigor Matemático**: Alto. Como el usuario es ingeniero matemático, no simplifiques las matemáticas. Desglosa las funciones de pérdida, las distribuciones latentes y las cotas de error (bounds).
- **Contradicciones/Evolución**: LeCun ha iterado sobre sus modelos. Si un paper nuevo descarta una técnica de uno anterior (ej. pasar de Contrastive a Non-Contrastive/VICReg), anótalo explícitamente y actualiza las páginas relevantes.
- **YAML Frontmatter**: All markdown files should start with YAML frontmatter for Dataview queries (e.g., `tags: [math, loss, jepa]`).

## Reglas de Mantenimiento (Linting/Maintenance)
Periodically check for:
- Orphan pages (pages with no incoming links).
- Missing definitions of math variables.
- Stale synthesis pages that don't include the latest ingested papers.

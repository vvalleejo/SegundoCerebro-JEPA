---
title: "V-JEPA 2.1"
tags: [entity, architecture, jepa, video, dense-features]
---

# V-JEPA 2.1

## Descripción
**V-JEPA 2.1** (2026) es una evolución directa de [[V-JEPA2]] diseñada por FAIR y la Universidad de Zaragoza. Su principal objetivo es extraer representaciones de grano fino (características densas) de videos e imágenes, superando una de las mayores debilidades históricas de las arquitecturas JEPA.

## Limitación de JEPAs Anteriores
En los modelos JEPA tradicionales (como I-JEPA o V-JEPA 2), la función de pérdida supervisa *exclusivamente* los parches enmascarados. Como resultado, el encoder no tiene incentivos para mantener la estructura espacial local de los parches de contexto visibles, usando sus tokens como meros acumuladores de información semántica global. Esto genera mapas de características (feature maps) ruidosos y fragmentados, perjudiciales para tareas densas como estimación de profundidad o segmentación.

## Solución V-JEPA 2.1
V-JEPA 2.1 incorpora:
1. **[[Dense_Predictive_Loss]]**: Añade una pérdida explícita sobre los tokens de contexto (Weighted Context Loss).
2. **Deep Self-Supervision**: Un MLP fusiona representaciones de múltiples niveles del encoder y calcula la pérdida en varias etapas, preservando la información estructural.
3. **Multi-Modal Tokenizers**: Combina tokens de imágenes estáticas y videos bajo un mismo pipeline de forma nativa.

## Enlaces Relacionados
- Paper: [[2026_V-JEPA2.1]]

---
title: "TC-JEPA (Text-Conditional JEPA)"
tags: [entity, architecture, jepa, multimodal, vision-language]
---

# TC-JEPA (Text-Conditional JEPA)

## Descripción
**TC-JEPA** (2026, Apple) es una arquitectura multimodal que extiende [[I-JEPA]] añadiendo condicionamiento textual de grano fino al módulo predictor.

## Arquitectura
1. **Context/Target Encoders**: Utilizan Vision Transformers (ViT) estándar, igual que I-JEPA.
2. **Predictor con Cross-Attention**: El predictor procesa los tokens de contexto visibles y las máscaras espaciales. Sin embargo, en múltiples capas intermedias, el predictor realiza *Cross-Attention* sobre los embeddings de palabras (tokens) de una descripción textual de la imagen (extraída vía un LLM como T5).
3. **Regularización de Esparsidad**: Para asegurar que los parches visuales presten atención solo a las palabras relevantes, se aplican penalizaciones de esparsidad (norma $L_1$) y consistencia sobre la matriz de similitud parche-palabra.

## Relación con otras JEPAs Multimodales
TC-JEPA es conceptualmente hermano de [[CHARM]] (que condiciona series temporales con texto) y una alternativa a [[MJEPA]] (que alinea audio y video usando un predictor cross-modal). TC-JEPA demuestra que el paradigma predictivo puede reemplazar totalmente al paradigma contrastivo (tipo CLIP) para entrenar modelos base Visión-Lenguaje, obteniendo representaciones que son semánticamente ricas pero que no pierden el detalle local.

## Enlaces Relacionados
- Paper: [[2026_TC-JEPA]]

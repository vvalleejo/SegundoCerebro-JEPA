---
title: "MJEPA (Multimodal JEPA for Audio-Visual)"
tags: [entity, architecture, jepa, multimodal, audio, video]
---

# MJEPA (Multimodal JEPA)

## Descripción
**MJEPA** (2026) es una arquitectura fundacional desarrollada por FAIR y NYU diseñada para aprender de manera simultánea e integrada de señales de audio y video utilizando la filosofía predictiva en el espacio latente.

## Arquitectura y Componentes
1. **Modality-Specific Tokenizers**: Para mantener la eficiencia, el modelo usa convoluciones 2D para espectrogramas de audio y convoluciones 3D para "tubelets" de video, proyectando ambas a la dimensión del transformer.
2. **Unified Shared Encoder**: A diferencia de los enfoques contrastivos clásicos (como CLIP), MJEPA procesa tanto los tokens visuales como los de audio a través del *mismo* Vision Transformer (ViT).
3. **Cross-Modal Predictors**: Consisten en MLPs ligeros que operan sobre la última capa del encoder. Toman la representación "promediada" de una modalidad (ej. audio) e intentan predecir la representación objetivo de la otra modalidad (ej. video).

## Importancia en el Doctorado
MJEPA demuestra cómo integrar múltiples canales perceptuales usando JEPAs. La técnica fundamental de "Cross-Modal Prediction" es lo que fuerza al espacio latente a alinear conceptos físicamente vinculados (el sonido de un cristal rompiéndose con la imagen del cristal roto). Es clave si se buscan World Models capaces de razonar mediante audio, visión, propiocepción o tacto en un mismo marco.

## Enlaces Relacionados
- Paper: [[2026_MJEPA]]

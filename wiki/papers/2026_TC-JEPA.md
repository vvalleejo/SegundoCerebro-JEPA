---
title: "Text-Conditional JEPA for Learning Semantically Rich Visual Representations"
authors: [Chen Huang, Xianhang Li, Vimal Thilak, Etai Littwin, Josh Susskind (Apple)]
year: 2026
tags: [paper, jepa, vision-language, multimodal, tc-jepa]
---

# Text-Conditional JEPA (TC-JEPA)

## Resumen Ejecutivo
Este paper (2026) presenta **TC-JEPA**, una variación de la arquitectura I-JEPA que aborda un problema inherente en la predicción latente: la **incertidumbre de predicción**. Si intentas predecir qué hay detrás de un parche oculto (ej. detrás de un perro en la imagen), la red tiene múltiples opciones plausibles (una pared, una estantería, un árbol). TC-JEPA soluciona esto introduciendo *condicionamiento de texto* usando los "captions" (descripciones) de las imágenes.

## Mecanismo de Condicionamiento
A diferencia de los modelos contrastivos (tipo CLIP) que alinean embeddings visuales y de texto globalmente, TC-JEPA lo hace de manera predictiva y de grano fino:
- Utiliza **Cross-Attention** sobre los tokens de texto dentro del predictor.
- Al predecir el bloque objetivo oculto, el predictor puede "leer" la descripción de la imagen (ej. "un perro sentado frente a una estantería"). Esto reduce la incertidumbre de la predicción y fuerza al espacio latente visual a alinearse con la semántica del lenguaje.

## Impacto
TC-JEPA establece un nuevo paradigma para el pre-entrenamiento de modelos Visión-Lenguaje (Vision-Language Models). En lugar de depender de pérdidas contrastivas, se basa exclusivamente en predicción de características. 
Empíricamente, demuestra superar a los métodos contrastivos (como SigLIP o CLIP) en tareas que requieren entendimiento visual de grano fino y razonamiento local, como la segmentación semántica y la detección de objetos, al mismo tiempo que escala excepcionalmente bien.

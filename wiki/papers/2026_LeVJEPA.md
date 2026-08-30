---
title: "LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics"
authors: [Lukas Kuhn, Lucas Maes, Giuseppe Serra, Quentin Le Lidec, Yann LeCun, Randall Balestriero, Florian Buettner]
year: 2026
tags: [video, jepa, self-supervised, collapse-free, sigreg]
---

# LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics

**Resumen**:
Este artículo presenta [[LeVJEPA]], el primer encoder de video entrenado con el objetivo libre de colapso de [[LeJEPA]], eliminando heurísticas tradicionales (target encoder, stop-gradient, y predictor network). Al utilizar la regularización [[SIGReg]] junto con una [[Invariance_Loss]] entre vistas globales y locales, el modelo garantiza matemáticamente que la representación no colapse.

**Contribuciones principales**:
- **Eficiencia computacional**: Reemplaza el enmascaramiento estructurado (tube masking) por un *uniform random token dropping* (descartando hasta el 95% de los tokens). Alcanza precisión similar a V-JEPA consumiendo menos de una quinta parte de los FLOPs de preentrenamiento.
- **Causalidad Temporal Integrada**: Al eliminar la asimetría de las ramas (ya no hay predictor), permite que la red se entrene con *block-causal attention* (atención bidireccional en el frame, y causal a través del tiempo), haciendo de la red una buena base para *streaming perception* o modelos autorregresivos.
- **Simplificación arquitectónica**: Solo requiere de un *Encoder* y un pequeño proyector. No hay EMA (Exponential Moving Average) y todo recae en un único hiperparámetro matemático.
- **Organización Semántica Emergente**: Aunque el modelo solo supervisa el token `[cls]` a nivel de clip, los tokens de los parches (no supervisados) adquieren de forma emergente una estructura densa y organizada semánticamente.

**Detalles de Arquitectura**: 
Ver [[2026_LeVJEPA_Arch]].

**Fundamentos Matemáticos**:
- La función de pérdida general está gobernada por: $\mathcal{L} = \mathcal{L}_{inv} + \lambda \mathcal{L}_{SIGReg}$
- Ver [[Invariance_Loss]] para el objetivo de predicción de embeddings de vista local hacia vista global.
- Ver [[SIGReg]] para el componente regularizador de distribuciones para evitar el colapso.

**Evaluación y Conclusiones**:
Alcanza paridad e incluso mejora respecto a baselines muy sólidos de video y compite casi a la par con DINOv2 (entrenado en imágenes de esos mismos videos) en evaluación centrada en apariencia (Appearance-centric), pero duplica su exactitud en métricas de entendimiento de movimiento (Motion-centric). Demuestra que el entrenamiento con video ya no está restringido a supercomputadoras; una ViT-Tiny puede entrenarse en una sola GPU de consumo en 12 horas.

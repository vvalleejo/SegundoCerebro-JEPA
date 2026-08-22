---
title: "Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations"
authors: [Yilun Kuang, Yash Dagade, Tim G. J. Rudner, Randall Balestriero, Yann LeCun]
year: 2026
tags: [paper, jepa, sparsity, regularization, math]
---

# Rectified LpJEPA: Representaciones Esparzas

## Resumen Ejecutivo
Este paper (2026) presenta **Rectified LpJEPA**, una evolución puramente matemática sobre el objetivo de regularización de los modelos JEPA. Aborda una deficiencia de métodos previos como [[LeWorldModel]] (que usa [[SIGReg]]): la regularización hacia distribuciones Gaussianas isotrópicas produce representaciones "densas" (todas las neuronas o dimensiones se activan un poco). Sin embargo, en neurociencia y deep learning, la **esparsidad** (sparsity) y la no negatividad son propiedades clave para lograr representaciones eficientes y desacopladas (disentangled).

## El Problema de SIGReg
SIGReg empuja al espacio latente a comportarse como una Gaussiana. Esto maximiza la entropía y evita el colapso, pero es incompatible con la esparsidad (norma $L_0$), donde la mayoría de los valores deberían ser exactamente cero.

## La Solución: Rectified Distribution Matching (RDMReg)
El paper introduce una nueva familia de distribuciones objetivo: **Rectified Generalized Gaussian (RGG)**. 
1. **RDMReg**: Es un nuevo término de regularización que reemplaza a SIGReg.
2. **Rectificación (ReLU)**: Aplica una rectificación (tipo ReLU) a los embeddings latentes.
3. **Matching Distribucional**: Alinea estas representaciones rectificadas hacia la distribución RGG utilizando proyecciones aleatorias y distancias de Wasserstein en 1D (Sliced Wasserstein Distance).

Al hacerlo, el modelo gana control analítico explícito sobre cuántos ceros (norma $L_0$) hay en la representación, manteniendo al mismo tiempo la propiedad de máxima entropía para los componentes no nulos.

## Resultados
Rectified LpJEPA aprende características esparzas y no negativas que superan en interpretabilidad y eficiencia a las representaciones densas, logrando resultados competitivos en tareas de clasificación de imágenes y demostrando que la esparsidad se puede controlar directamente a través del diseño de la distribución objetivo en JEPAs.

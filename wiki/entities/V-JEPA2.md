---
title: "V-JEPA 2 & V-JEPA 2-AC"
tags: [entity, architecture, jepa, video, action-conditioned]
---

# V-JEPA 2 y V-JEPA 2-AC

## Descripción
**V-JEPA 2 (Video Joint-Embedding Predictive Architecture 2)** es un modelo fundacional de visión (FAIR, 2025) entrenado para entender representaciones espacio-temporales en video mediante predicción de enmascaramiento latente.

## Componentes Clave
1. **V-JEPA 2 (Base)**: Encoder ViT pre-entrenado en 1M de horas de video con el objetivo de mask-denoising en el espacio de representación (sin reconstrucción de píxeles). A diferencia de [[LeWorldModel]], depende de la asimetría arquitectónica (EMA y Stop-Gradients) para evitar el colapso.
2. **V-JEPA 2-AC (Action-Conditioned)**: Es la extensión "World Model". Consiste en un predictor Transformer auto-regresivo entrenado sobre los embeddings congelados de V-JEPA 2. Se condiciona mediante las acciones de control del robot y los estados propioceptivos para predecir el siguiente estado latente.

## Uso en Planificación (Model Predictive Control)
La arquitectura se utiliza para planificación robótica *Zero-Shot* optimizando una secuencia de acciones $\hat{a}_{1:T}$ que minimiza la distancia latente a un objetivo:

$$ \mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) := \| P_\phi(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1 $$

Donde $z_k$ es el estado visual inicial, $s_k$ es el estado del robot, y $z_g$ es el estado latente objetivo. La optimización se realiza muestreando acciones a través del **Cross-Entropy Method (CEM)**.

## Enlaces Relacionados
- Paper: [[2025_V-JEPA2]]

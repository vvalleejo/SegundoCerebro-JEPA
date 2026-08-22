---
title: "Variational JEPA (VJEPA)"
tags: [entity, architecture, jepa, probabilistic]
---

# Variational JEPA (VJEPA)

## Descripción
**Variational JEPA (VJEPA)** es una generalización de la familia de arquitecturas [[JEPA]] introducida por Yongchao Huang en 2026. Transforma el paradigma determinista de predicción latente en un modelo predictivo explícitamente **probabilístico**.

## Diferencias con JEPA Determinista
A diferencia de modelos como [[LeWorldModel]] o I-JEPA que predicen un estado latente único $\hat{z}_{t+1}$, VJEPA aprende una distribución predictiva $p_\phi(Z_T \mid Z_C, \xi_T)$ sobre los posibles estados latentes futuros. Esto le otorga varias ventajas críticas:
1. **Estimación de Incertidumbre**: El predictor no solo emite una media, sino también una varianza (o matriz de covarianza), capturando la incertidumbre aleatoria (aleatoric uncertainty) del entorno.
2. **Futuros Multimodales**: Puede modelar bifurcaciones en el futuro sin promediarlas hacia un estado "irreal" (problema común con la pérdida MSE determinista).

## Arquitectura
1. **Context Encoder ($f_\theta$)**: Mapea el historial/contexto a un embedding determinista $Z_C$.
2. **Target Encoder ($f_{\theta'}$)**: Actúa como el modelo de inferencia amortizado, produciendo una distribución $q_{\theta'}(Z_T \mid x_T)$ (por ejemplo, mediante una Gaussiana parametrizada).
3. **Probabilistic Predictor ($p_\phi$)**: Genera la distribución predictiva usando el contexto y la información estructural $\xi_T$.

## Relación con BJEPA
VJEPA es un caso especial de su extensión natural, [[BJEPA]] (Bayesian JEPA), cuando se utiliza un prior uniforme/no informativo para la tarea.

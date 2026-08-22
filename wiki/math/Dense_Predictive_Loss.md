---
title: "Dense Predictive Loss"
tags: [math, loss, v-jepa, dense-features]
---

# Dense Predictive Loss ($\mathcal{L}_{dense}$)

Para solucionar el problema de fragmentación espacial en las representaciones de los modelos JEPA, [[V-JEPA2.1]] introduce una pérdida que supervisa tanto los parches visibles (contexto) como los enmascarados (targets).

## Formulación

La pérdida total se define como:
$$ \mathcal{L}_{dense} = \mathcal{L}_{predict} + \mathcal{L}_{ctx} $$

1. **Prediction Loss ($\mathcal{L}_{predict}$)**:
Es la pérdida estándar de V-JEPA aplicada solo a los tokens enmascarados.
$$ \mathcal{L}_{predict} = \frac{1}{|M|} \sum_{i \in M} \| P_\phi(E_\theta(x), \Delta_y)_i - \text{sg}(E_{\bar{\theta}}(y)_i) \|_1 $$
*(Donde $M$ son los índices de los parches enmascarados, $P_\phi$ el predictor, y $\text{sg}$ es el stop-gradient).*

2. **Weighted Context Loss ($\mathcal{L}_{ctx}$)**:
Se aplica sobre los tokens visibles (el contexto $C$), obligándoles a predecir también las características locales de su propio parche.
$$ \mathcal{L}_{ctx} = \frac{1}{|C|} \sum_{i \in C} \lambda_i \| P_\phi(E_\theta(x), \Delta_y)_i - \text{sg}(E_{\bar{\theta}}(y)_i) \|_1 $$

### Ponderación Dinámica ($\lambda_i$)
Aplicar $\mathcal{L}_{ctx}$ de forma ingenua perjudica la comprensión global del modelo. Por ello, se pondera dinámicamente según la distancia espacio-temporal del parche de contexto al parche enmascarado más cercano:
$$ \lambda_i = \frac{\lambda}{\sqrt{d_{min}(i, M)}} $$

Donde $d_{min}$ es la distancia (en número de bloques) entre el token de contexto $i$ y el token enmascarado más cercano en $M$. Esto fuerza la continuidad local en las regiones fronterizas entre lo visible y lo oculto, generando características densas y coherentes, esenciales para robótica de precisión.

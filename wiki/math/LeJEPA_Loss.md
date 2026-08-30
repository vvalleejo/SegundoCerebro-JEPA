---
title: "LeJEPA Loss Objective"
tags: [math, loss, jepa, lejepa, ssl]
---

# LeJEPA Loss Objective ($\mathcal{L}_{\text{LeJEPA}}$)

El objetivo de entrenamiento de [[LeJEPA]] unifica la predicción de invariancia multi-vista con la regularización anti-colapso [[SIGReg]], parametrizado por un único factor de balance $\lambda \in [0, 1]$:

$$
\mathcal{L}_{\text{LeJEPA}}(\{x_{n,v}\}_{n,v=1}^{B,V}) \triangleq \lambda \frac{1}{V} \sum_{v=1}^V \text{SIGReg}(\{z_{n,v}\}_{n=1}^B) + \frac{1 - \lambda}{B} \sum_{n=1}^B \mathcal{L}_{\text{pred}}^{(V_g)}(\{z_{n,v}\}_{v=1}^V)
$$

---

## 1. Término de Predicción de Invariancia ($\mathcal{L}_{\text{pred}}^{(V_g)}$)

Dado un conjunto de $V = V_g + V_l$ vistas de una misma muestra $x_n$ (donde las primeras $V_g$ corresponden a vistas globales sin distorsión severa y las restantes $V_l$ a vistas locales con recortes agresivos):

$$
\mathcal{L}_{\text{pred}}^{(V_g)}(\{z_{n,v}\}_{v=1}^V) \triangleq \frac{1}{V_g} \sum_{v=1}^{V_g} \frac{1}{V} \sum_{v'=1}^V \|z_{n,v} - z_{n,v'}\|_2^2
$$

### Equivalencia con el Centroide Global ($\mu_n$)
Reordenando los términos algebraicamente:

$$
\mathcal{L}_{\text{pred}}^{(V_g)}(\{z_{n,v}\}_{v=1}^V) = \frac{1}{V} \sum_{v'=1}^V \left\| \left( \frac{1}{V_g} \sum_{v=1}^{V_g} z_{n,v} \right) - z_{n,v'} \right\|_2^2 = \frac{1}{V} \sum_{v'=1}^V \|\mu_n - z_{n,v'}\|_2^2
$$

Donde $\mu_n \triangleq \frac{1}{V_g}\sum_{v=1}^{V_g} z_{n,v}$ es el centroide de los embeddings de las vistas globales. 

---

## 2. Término de Regularización ([[SIGReg]])

Para evitar el colapso a una representación trivial constante o de dimensionalidad reducida sin requerir redes *teacher*, *stop-gradients* ni *predictores*:

$$
\text{SIGReg}(\{z_{n,v}\}_{n=1}^B) \triangleq \frac{1}{M} \sum_{m=1}^M T(a_m^\top z_{\cdot, v})
$$

Donde $a_m \in \mathbb{S}^{K-1}$ son $M$ direcciones unitarias aleatorias remuestreadas en cada paso de gradiente, y $T$ es el estadístico de **Epps-Pulley** evaluado contra la Gaussiana estándar $\mathcal{N}(0, 1)$.

---

## 3. Propiedades Clave

1. **Unicidad de Hiperparámetro**: A diferencia de métodos con múltiples ponderaciones (e.g., VICReg con $\lambda_{inv}, \mu_{var}, \nu_{cov}$), LeJEPA utiliza un único hiperparámetro $\lambda$ cuyo valor por defecto ($\lambda = 0.05$) es universalmente aplicable a través de diversas arquitecturas y dominios.
2. **Caso Límite y Recuperación de VICReg**: Si se sustituye la estadística de Epps-Pulley por un test degenerado que solo evalúa la media y la varianza marginal ($T(\{x_n\}) = \text{mean}(x)^2 + (\text{std}(x) - 1)^2$), el objetivo de LeJEPA recupera analíticamente el método **VICReg** en el límite de infinitas direcciones ($M \to \infty$). Sin embargo, igualar únicamente los momentos de segundo orden resulta insuficiente para descartar atajos degenerados de distribución (Teorema de Insuficiencia de $K$ Momentos).
3. **Escalabilidad de Gradientes**: Los gradientes respecto a los parámetros del encoder están uniformemente acotados, garantizando estabilidad numérica sin necesidad de *schedulers* complejos para la tasa de aprendizaje o el *weight decay*.

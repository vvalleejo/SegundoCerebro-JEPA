---
title: "LeJEPA Loss Objective"
type: "math"
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

### Relación con el centroide global ($\mu_n$)
Sea $\mu_n \triangleq \frac{1}{V_g}\sum_{v=1}^{V_g} z_{n,v}$ el centroide de los embeddings de las vistas globales. Para cada $v'$ fijo, la descomposición sesgo-varianza da:

$$
\frac{1}{V_g}\sum_{v=1}^{V_g}\|z_{n,v}-z_{n,v'}\|_2^2 = \|\mu_n - z_{n,v'}\|_2^2 + \underbrace{\frac{1}{V_g}\sum_{v=1}^{V_g}\|z_{n,v}-\mu_n\|_2^2}_{\sigma^2_{g,n}\ \text{(no depende de } v')}
$$

Promediando sobre $v'$:

$$
\mathcal{L}_{\text{pred}}^{(V_g)} = \frac{1}{V} \sum_{v'=1}^V \|\mu_n - z_{n,v'}\|_2^2 \;+\; \sigma^2_{g,n}
$$

> [!important] Corrección (arbitraje con el PDF)
> El paper escribe las Eqs. 5–7 (p. 12) como una cadena de igualdades. **Tal como están escritas no son iguales:** la forma "a pares" (Eq. 5) excede a la forma "centroide" (Eq. 7) en la dispersión intra-globales $\sigma^2_{g,n}$. Ambas tienen el mismo mínimo (todas las vistas iguales), pero sus gradientes difieren. El **código oficial implementa la forma centroide (Eq. 7)**: `centers = g_emb.mean(0); sim = (centers - a_emb).square().mean()` (Algorithm 1). Además, $\mu_n$ **no** lleva stop-gradient (p. 12). La bóveda adopta la Eq. 7 como definición canónica.

---

## 2. Término de Regularización ([[SIGReg]])

Para evitar el colapso a una representación trivial constante o de dimensionalidad reducida sin requerir redes *teacher*, *stop-gradients* ni *predictores*:

$$
\text{SIGReg}(\{z_{n,v}\}_{n=1}^B) \triangleq \frac{1}{M} \sum_{m=1}^M T(a_m^\top z_{\cdot, v})
$$

Donde $a_m \in \mathbb{S}^{K-1}$ son $M$ direcciones unitarias aleatorias remuestreadas en cada paso de gradiente, y $T$ es el estadístico de **Epps-Pulley** evaluado contra la Gaussiana estándar $\mathcal{N}(0, 1)$. SIGReg se aplica **por vista** y se promedia sobre las $V$ vistas. Definición completa y detalles numéricos en [[SIGReg]].

*Notación:* aquí $K$ es la dimensión del embedding y $M$ el número de direcciones, como en el paper. En la notación canónica de la bóveda son $D$ y $K$, respectivamente (ver `GEMINI.md` §5).

---

## 3. Propiedades Clave

1. **Unicidad de Hiperparámetro**: A diferencia de métodos con múltiples ponderaciones (e.g., VICReg con $\lambda_{inv}, \mu_{var}, \nu_{cov}$), LeJEPA utiliza un único hiperparámetro $\lambda$ en **forma convexa**. Los autores recomiendan $\lambda = 0.05$ "as a robust default", junto con $V_g=2$, $V_l=8$ y batch $\ge 128$ (PDF p. 13). Ojo: los Experiment Details (p. 14) usan $V=8$ con $V_g=2$, así que el propio paper es inconsistente en el número de vistas locales. Este $\lambda$ **no es comparable** con el de LeVJEPA ($0.02$) ni con el de LeWM ($0.1$), porque ambos usan la forma aditiva $\mathcal L_{\text{pred}}+\lambda\,\text{SIGReg}$.
2. **Caso Límite y Recuperación de VICReg**: Si se sustituye la estadística de Epps-Pulley por un test degenerado que solo evalúa la media y la varianza marginal ($T(\{x_n\}) = \text{mean}(x)^2 + (\text{std}(x) - 1)^2$), el objetivo de LeJEPA recupera analíticamente el método **VICReg** en el límite de infinitas direcciones ($M \to \infty$). Sin embargo, igualar únicamente los momentos de segundo orden resulta insuficiente para descartar atajos degenerados de distribución (Teorema de Insuficiencia de $K$ Momentos).
3. **Escalabilidad de Gradientes**: Los gradientes respecto a los parámetros del encoder están uniformemente acotados, garantizando estabilidad numérica sin necesidad de *schedulers* complejos para la tasa de aprendizaje o el *weight decay*.

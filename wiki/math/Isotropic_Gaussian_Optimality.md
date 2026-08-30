---
title: "Isotropic Gaussian Optimality for Foundation Models"
tags: [math, theory, probability, lejepa, statistics]
---
# Optimalidad de la Distribución Gaussiana Isotrópica

Uno de los aportes teóricos fundamentales de [[2025_LeJEPA]] (Balestriero & LeCun, 2025) es responder rigurosamente a la pregunta: **¿Qué distribución de probabilidad deben seguir los embeddings $Z = f_\theta(X) \in \mathbb{R}^K$ en un modelo fundacional para minimizar el riesgo de error en tareas *downstream* arbitrarias?**

Se demuestra que, bajo restricciones de covarianza escalar fijas, la **distribución Gaussiana isotrópica** $\mathcal{N}(0, \sigma^2 I_K)$ es la única que minimiza el sesgo y la varianza para sondas lineales, sondas $k$-NN no paramétricas y métodos de regresión por kernel.

---

## 1. Sondas Lineales (Linear Probing)

Consideremos una matriz de $N$ embeddings $Z \in \mathbb{R}^{N \times K}$ y un vector desconocido de etiquetas $y \in \mathbb{R}^N$. El estimador lineal de mínimos cuadrados regularizado (Ridge Regression / Tikhonov) está dado por:

$$
\hat{\beta} = \arg\min_{\beta \in \mathbb{R}^K} \|y - Z\beta\|_2^2 + \lambda_{\text{wd}}\|\beta\|_2^2 = (Z^\top Z + \lambda_{\text{wd}} I)^{-1} Z^\top y

$$

Sean dos geometrías de embedding con la misma traza de covarianza $\bar{\lambda} = \frac{1}{K}\sum_{k=1}^K \lambda_k$:

- $Z_{\text{iso}}$: con autovalores de covarianza idénticos $\lambda_k = \bar{\lambda}, \forall k$.
- $Z_{\text{aniso}}$: con autovalores de covarianza distintos $\lambda_K > \lambda_1$.

### Lema 1: La Anisotropía Amplifica el Sesgo

Para cualquier matriz de covarianza anisotrópica con $\lambda_K > \lambda_1$ y $\lambda_{\text{wd}} > 0$, siempre existe una tarea *downstream* (vector de etiquetas $y$) tal que el estimador sobre $Z_{\text{aniso}}$ tiene mayor sesgo que sobre $Z_{\text{iso}}$:

$$
\|\text{Bias}(\hat{\beta})\|_{\text{non-isotropic}} = \frac{\lambda_{\text{wd}}}{\lambda_p + \lambda_{\text{wd}}}\|\beta_{\text{true}}\| > \frac{\lambda_{\text{wd}}}{\bar{\lambda} + \lambda_{\text{wd}}}\|\beta_{\text{true}}\| = \|\text{Bias}(\hat{\beta})\|_{\text{isotropic}}

$$

Donde $\lambda_p < \bar{\lambda}$ es el menor autovalor de la matriz de covarianza.

### Lema 2: La Anisotropía Amplifica la Varianza

Para $\lambda_{\text{wd}} = 0$, la varianza total del estimador $\hat{\beta}$ se minimiza estrictamente cuando los embeddings son isotrópicos:

$$
\text{tr}(\text{Var}(\hat{\beta}_{\text{aniso}})) = \sigma^2 \sum_{j=1}^K \frac{1}{\lambda_j} > \sigma^2 \sum_{j=1}^K \frac{1}{\bar{\lambda}} = \text{tr}(\text{Var}(\hat{\beta}_{\text{iso}}))

$$

Por la desigualdad de Jensen aplicada a la función estrictamente convexa $f(x) = 1/x$.

---

## 2. Sondas No Lineales: $k$-NN Radial

Para una sonda $k$-NN basada en radio $r_0$ con densidad latente $p_z \in \mathcal{C}^3$ y función objetivo $\eta(z) = \mathbb{E}[Y | z]$:

### Teorema (Optimalidad en $k$-NN):

El sesgo cuadrático integrado (*Integrated Squared Bias*, ISB) sobre el espacio de consultas satisface:

$$
\mathbb{E}_z [\text{Bias}(z)^2] = \frac{r_0^4}{(K + 2)^2} \tau_g^2 J(p) + \mathcal{O}(r_0^4)

$$

Donde $J(p)$ es el **funcional de información de Fisher**:

$$
J(p) \triangleq \int_{\mathbb{R}^K} \|\nabla \log p(x)\|_2^2 p(x) dx

$$

Y la distribución Gaussiana isotrópica es la **única minimizadora** de $J(p)$ entre todas las distribuciones con covarianza acotada.

---

## 3. Sondas por Métodos de Kernel (Nadaraya-Watson)

Para un estimador kernel Nadaraya-Watson con ancho de banda $h$:

### Teorema (Optimalidad en Regresión por Kernel):

$$
\sup_{m \in \mathcal{M}(L, B)} \mathbb{E}_z [\text{Bias}(\hat{y}(z))^2] \le \left( \frac{h^2 \mu_2(K)}{2} \right)^2 \left( 2B^2 + 8L^2 J(p) \right) + o(h^4)

$$

Bajo cualquier restricción escalar sobre la matriz de covarianza $\Sigma$ (traza $\text{tr}(\Sigma) = t$, determinante $\det(\Sigma) = \delta$, norma Frobenius $\|\Sigma\|_F = c$, o radio espectral $\rho(\Sigma) \le r$), la densidad que minimiza la cota superior del sesgo integrado es unívocamente la **Gaussiana isotrópica** $\mathcal{N}(0, s I_d)$ con:

- Traza $t$: $s = t/d$
- Determinante $\delta$: $s = \delta^{1/d}$
- Norma Frobenius $c$: $s = c/\sqrt{d}$
- Radio espectral $r$: $s = r$

---

## Conclusión

La imposición de una distribución Gaussiana isotrópica en el espacio latente (mediante regularizadores como [[SIGReg]]) no es una heurística ad-hoc, sino una **condición matemáticamente necesaria y óptima** para garantizar la mínima pérdida esperada en tareas futuras no observadas durante el preentrenamiento.

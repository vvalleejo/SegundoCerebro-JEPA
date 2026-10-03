---
title: "Sketched-Isotropic-Gaussian Regularizer (SIGReg)"
type: "math"
tags: [math, regularization, loss, anti-collapse, lejepa, statistics]
---

# Sketched-Isotropic-Gaussian Regularizer (SIGReg)

## Motivación
En las arquitecturas Joint-Embedding Predictive Architectures (JEPA), predecir el futuro estado latente utilizando solo pérdidas de distancia (como MSE) conduce al **colapso de la representación**, donde el encoder mapea todas las entradas a una constante trivial. 

Para resolver este problema sin recurrir a heurísticas (como redes *teacher* con EMA, *stop-gradient*, o capas de *whitening* de complejidad cuadrática), Balestriero y LeCun (2025, [[2025_LeJEPA]]) introdujeron **SIGReg**. Basado en la prueba de que la distribución Gaussiana isotrópica $\mathcal{N}(0, I_K)$ es la óptima para minimizar el riesgo downstream ([[Isotropic_Gaussian_Optimality]]), SIGReg fuerza a los embeddings a seguir dicha distribución mediante un contraste de hipótesis estadísticas en proyecciones aleatorias.

---

## Formulación Matemática

Sea $Z = \{z_n\}_{n=1}^N \subset \mathbb{R}^K$ el lote de $N$ embeddings producidos por el encoder $f_\theta$.

### 1. Descomposición por Teorema de Cramér-Wold
Evaluar la normalidad multivariada directamente en alta dimensión es intratable computacionalmente. SIGReg utiliza el **Teorema de Cramér-Wold**, según el cual:
$$
Z \overset{d}{=} \mathcal{N}(0, I_K) \iff \langle a, Z \rangle \overset{d}{=} \mathcal{N}(0, 1), \quad \forall a \in \mathbb{S}^{K-1}
$$

En cada iteración de optimización, se muestrean $M$ direcciones aleatorias $A = \{a_1, \ldots, a_M\}$ uniformemente sobre la hiperesfera unidad $\mathbb{S}^{K-1}$.

### 2. Estadístico de Epps-Pulley en 1D
Para cada dirección $a \in A$, se evalúa la normalidad univariada de la proyección $\{a^\top z_n\}_{n=1}^N$ mediante la prueba de **Epps-Pulley (1983)**, que compara la **Función Característica Empírica (ECF)** $\hat{\phi}_N(t) = \frac{1}{N}\sum_{n=1}^N e^{i t a^\top z_n}$ con la función característica teórica de la normal estándar $\phi_0(t) = e^{-t^2/2}$:

$$
\text{SIGReg}(A, Z) \triangleq \frac{1}{|A|} \sum_{a \in A} N \int_{-\infty}^{\infty} \left| \hat{\phi}_N(t; a^\top Z) - e^{-t^2/2} \right|^2 e^{-t^2/2} dt
$$

La integral se aproxima con **cuadratura trapezoidal determinista**. En LeJEPA se usan 17 nodos en $[-5,5]$ (Algorithm 1: `t = torch.linspace(-5, 5, 17)`). Los autores señalan que la simetría del integrando permite duplicar los nodos "for free" (PDF §5.1).

> [!important] Qué dice exactamente el PDF de LeJEPA (arbitraje)
> - **Factor $N$:** es el tamaño de batch **global**. En el código, `N = x.size(0) * world_size` (Algorithm 1, p. 10).
> - **Peso $w(t)$:** hay una inconsistencia entre texto y código. El texto (p. 9) dice $w(t)=e^{-t^2/\sigma^2}$ con $\sigma=1$. El código usa `exp(-0.5 t²)` como función característica objetivo **y** como ventana, es decir, $w(t)=e^{-t^2/2}$. La fórmula de arriba sigue al código.
> - **Direcciones:** se remuestrean **en cada paso**; el generador se siembra con `global_step`. Algorithm 1 tiene `num_slices=256` por defecto, pero el texto recomienda empezar con 1024 (p. 13).
> - **Máximo frente a media:** Definición 2 (p. 7) sustituye el $\max_a$ del teorema por una media sobre $A$ "to avoid sparse gradient".

### 3. Variantes de implementación entre papers
El nombre es el mismo, pero los detalles numéricos **no**. Los valores de $\lambda$ no son comparables entre papers.

| Paper | Estadístico | Factor $N$ | Nodos / dominio | Direcciones | Forma de $\lambda$ | $\lambda$ |
|---|---|---|---|---|---|---|
| [[wiki/papers/2025_LeJEPA\|LeJEPA]] | Epps–Pulley | sí (batch global) | 17 en $[-5,5]$ | 256 (código) / 1024 (texto) | convexa | 0.05 |
| [[wiki/papers/2026_LeVJEPA\|LeVJEPA]] | ECF, Eq. 3 | no | 17 en $[0,3]$ | 1024 | aditiva | 0.02 |
| [[wiki/papers/2026_LeWorldModel\|LeWM]] | Epps–Pulley | no | $T$ uniformes en $[0.2,4]$ (nº no fijado) | 1024 | aditiva | 0.1 (bisección) |
| [[wiki/papers/2026_Semigroup-JEPA\|SG-JEPA]] | ECF, Eq. 160 | sí ($B$) | 17 knots | 1024 | aditiva | 0.09–0.72 según tarea |
| [[wiki/papers/2026_SkyJEPA\|SkyJEPA]] | Epps–Pulley sobre latentes **predichos** del rollout | — | 17 knots | — | aditiva | 0.02 |
| [[wiki/papers/2026_MotionJEPA\|MotionJEPA]] | Epps–Pulley (p. 3), sobre $z$ y sobre $d$ | — | config. LeWM | — | pesos fijos | $\lambda_z=0.25,\ \lambda_d=2$ |
| [[wiki/papers/2026_AD-E2E-JEPA\|AD-E2E-JEPA]] | Epps–Pulley **por posición de parche** y paso temporal, sobre embeddings proyectados | — (no se indica) | 17 en $[0,3]$ | 1024 | aditiva | 0.09 (0.025 con batch 512) |

Fuentes: LeVJEPA App. A (p. 12); LeWM App. A (p. 13), donde además llama $\lambda$ al ancho de banda $w(t)=e^{-t^2/(2\lambda^2)}$, en conflicto con el peso de la pérdida; SG-JEPA App. C y H.8; SkyJEPA §IV-B (p. 4–5); AD-E2E-JEPA Eq. 10, App. A.1, Tab. 1.

> [!important] Variante parche a parche (AD-E2E-JEPA)
> LeWM aplica SIGReg al CLS global en cada paso temporal y promedia en el tiempo. AD-E2E-JEPA lo aplica **por separado a cada posición** $l$ de la rejilla espacio-temporal proyectada ($N=K+2$ frames × $H'W'=32$ posiciones), sobre el batch, y promedia (Eq. 10):
> $$\mathcal L_{\text{SIGReg}}=\frac{1}{NH'W'M}\sum_{l=1}^{NH'W'}\sum_{m=1}^{M}T\Big(\big\{\langle z_{l,b},u^{(m)}\rangle\big\}_{b=1}^{B}\Big),\quad z_{l,b}\in\mathbb R^{D},\ u^{(m)}\in\mathbb S^{D-1},\ D=256$$
> Es un test **marginal**: cada parche, visto a lo largo del batch, debe parecer $\mathcal N(0,I_D)$, pero no se impone independencia entre parches. En la notación del paper, $T(\cdot)$ es el test (no el nº de nodos) y $M$ las direcciones (canónico $K$).

> [!warning] Descripciones erróneas corregidas en la bóveda
> SIGReg **no** es un test de Cramér–von Mises ni una penalización de media, varianza y kurtosis. Los papers usan Epps–Pulley sobre la función característica empírica y rechazan explícitamente los tests basados en momentos (ver §1 abajo). El pariente más cercano es **RDMReg** ([[RDMReg]]). También es una instancia de Cramér–Wold, pero usa un test **two-sample** (sliced $W_2$) en lugar de uno *one-sample*. LpWM llega a reimplementar el baseline LeWM con SWD (LpWM App. H.1).

---

## Propiedades Teóricas y Computacionales

### 1. Gradientes Acotados y Estabilidad (Teorema 4)
A diferencia de los métodos basados en momentos (ej. *Jarque-Bera*, *kurtosis* o *skewness*, cuyos gradientes crecen polinómicamente y explotan con valores atípicos), el estadístico de Epps-Pulley opera sobre exponenciales complejas acotadas ($|e^{itx}| = 1$). Las derivadas primera y segunda satisfacen:
$$
\left| \frac{\partial \text{EP}(a)}{\partial z_i} \right| \le \frac{4\sigma^2}{N}, \quad \left| \frac{\partial^2 \text{EP}(a)}{\partial z_i^2} \right| \le \frac{C \sqrt{\pi}\sigma^3}{2N}
$$
Garantizando gradientes Lipschitz estables e insensibles a outliers.

### 2. Venciendo la Maldición de la Dimensionalidad (Teorema 5)
Bajo una regularidad de Sobolev $\alpha$ de la densidad del encoder $p_\theta \in H^\alpha(\mathbb{R}^K)$, el error de interpolación esférica decrece a una tasa de:
$$
\mathcal{O}\left( |A|^{-\frac{2\alpha}{K-1}} \right)
$$
Dado que el remuestreo de direcciones aleatorias en cada minibatch mediante SGD tiene un efecto acumulativo lineal con el número de pasos, un número reducido de cortes ($M = 16 \text{ a } 256$) es suficiente para restringir estrictamente el espacio de alta dimensión.

### 3. Complejidad Lineal $\mathcal{O}(N)$ y Compatibilidad DDP
- **Complejidad:** proyectar cuesta $\mathcal{O}(N K M)$ y evaluar la ECF en $T$ nodos cuesta $\mathcal{O}(N M T)$. En total es **lineal en $N$ y en $K$**, sin matriz $K\times K$. El paper lo resume como "linear memory and computational complexity of $\mathcal O(N)$" (p. 10) y "linear in dimension and sample size" (p. 2).
- **Sesgo de minibatch** (Theorem 6, p. 12): $\mathcal O(1/N)$ en la pérdida y en el gradiente. Según los autores es "minimal … even for minibatches as small as 16".
- **DDP (Distributed Data Parallel)**: Como la ECF es un promedio simple de exponenciales complejas, la agregación entre múltiples GPUs se realiza con una única operación `all_reduce` sobre un tensor de tamaño $2 \times M \times T$, con costo de comunicación independiente de $N$ y de $K$.

---

## Referencias en la Wiki
- [[2025_LeJEPA]]: Artículo fundacional que introduce SIGReg y demuestra la optimalidad de la Gaussiana.
- [[Isotropic_Gaussian_Optimality]]: Demostración matemática del porqué la normal isotrópica minimiza el riesgo downstream.
- [[LeJEPA_Loss]]: Formulación combinada con la pérdida de predicción / invariancia.
- [[2026_LeWorldModel]]: Uso de SIGReg para modelos de mundo latentes *end-to-end*.
- [[2026_LeVJEPA]]: Aplicación de SIGReg a video con atención causal y *token dropping*.

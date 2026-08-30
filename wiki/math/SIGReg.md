---
title: "Sketched-Isotropic-Gaussian Regularizer (SIGReg)"
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

La integral se aproxima numéricamente mediante cuadratura trapezoidal determinista con $T \approx 17$ puntos en el intervalo $[-5, 5]$ (o $[0, 3]$ aprovechando simetría).

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
- **Complejidad**: $\mathcal{O}(N \cdot K \cdot M)$, lineal en tamaño de batch $N$ y dimensión $K$.
- **DDP (Distributed Data Parallel)**: Como la ECF es un promedio simple de exponenciales complejas, la agregación entre múltiples GPUs se realiza con una única operación `all_reduce` sobre un tensor de tamaño $2 \times M \times T$, con costo de comunicación independiente de $N$ y de $K$.

---

## Referencias en la Wiki
- [[2025_LeJEPA]]: Artículo fundacional que introduce SIGReg y demuestra la optimalidad de la Gaussiana.
- [[Isotropic_Gaussian_Optimality]]: Demostración matemática del porqué la normal isotrópica minimiza el riesgo downstream.
- [[LeJEPA_Loss]]: Formulación combinada con la pérdida de predicción / invariancia.
- [[2026_LeWorldModel]]: Uso de SIGReg para modelos de mundo latentes *end-to-end*.
- [[2026_LeVJEPA]]: Aplicación de SIGReg a video con atención causal y *token dropping*.

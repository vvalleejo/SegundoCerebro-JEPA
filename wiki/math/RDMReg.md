---
title: "Rectified Distribution Matching Regularization (RDMReg)"
tags: [math, regularization, sparsity, jepa]
---

# Rectified Distribution Matching Regularization (RDMReg)

## 1. Motivación y Contexto

En el aprendizaje autosupervisado y las arquitecturas JEPA, evitar el colapso de dimensionalidad (donde todas las representaciones convergen a un punto constante) es crítico. Enfoques como [[SIGReg]] utilizan el Teorema de Cramér-Wold para igualar proyecciones 1D de latentes a una Gaussiana isotrópica estándar, lo que promueve el uso eficiente de todo el espacio, resultando en **representaciones densas** (donde todos los valores latentes son no nulos).

**RDMReg** se introduce en **LpWM** ([[2026_LpWM]]) para forzar **representaciones dispersas (sparse) y no negativas**. La intuición es que la esparcidad (o una aproximación relajada a una representación *one-hot*) ayuda a linealizar las dinámicas subyacentes, haciendo más simple la predicción y planificación.

## 2. Formulación Matemática

Sea $Z \in \mathbb{R}^{d}$ el vector de embedding. RDMReg minimiza la distancia entre las distribuciones marginales proyectadas 1D de las características y las proyecciones 1D de una distribución objetivo rectificada.

### Definición de RDMReg
La regularización se formula como la esperanza sobre proyecciones aleatorias uniformes:

$$ \mathcal{R}_{\text{RDMReg}}(Z) = \mathbb{E}_{c \sim \text{Unif}(\mathbb{S}^{d-1}_2)} \left[ \mathcal{L}\left(\mathbb{P}_{c^\top Z} \,\|\, \mathbb{P}_{c^\top Y}\right) \right] $$

Donde:
- $c \in \mathbb{S}^{d-1}_2$ es un vector de proyección aleatorio en la hiperesfera unidad $L_2$.
- $\mathbb{P}_{X}$ denota la distribución de probabilidad de la variable aleatoria $X$.
- $\mathcal{L}(\cdot \| \cdot)$ es una métrica de discrepancia entre distribuciones. En RDMReg se implementa utilizando la distancia **2-Wasserstein**.
- $Y$ es un vector aleatorio objetivo muestreado de una distribución Rectificada:
  
$$ Y \sim \prod_{i=1}^d \text{ReLU}\left(\text{GN}_p(\mu, \sigma)\right) $$

### La Distribución Objetivo: Gaussiana Generalizada (RGG)
$\text{GN}_p(\mu, \sigma)$ es la distribución Gaussiana Generalizada, caracterizada por la norma $L_p$. Al fijar:
- $\mu = 0$
- $\sigma = \sqrt{1/2}$
- $p = 1$

La distribución se reduce a una **Distribución de Laplace Rectificada**. Al aplicar la función $\text{ReLU}$ sobre esta distribución, la probabilidad concentra una masa finita (aprox. 50% si está centrada) exactamente en cero, lo que induce una **esparcidad estricta e independiente** a lo largo de cada dimensión del código latente.

## 3. Regularización de Jaccard Temporal (TJ)

Al modelar dinámicas físicas, es deseable que el **soporte** (las dimensiones no nulas) del código disperso identifique el régimen dinámico actual (ej. interacciones de contacto), y cambie solo cuando el régimen cambia.

Para forzar esta coherencia semántica en transiciones en el tiempo, LpWM añade una pérdida basada en el **Índice Jaccard** entre pasos de tiempo adyacentes:

$$ \mathcal{J}(z_t, z_{t+1}) = \frac{\sum_{i=1}^d \mathbb{1}\{z_{t,i} > 0 \land z_{t+1,i} > 0\}}{\sum_{i=1}^d \mathbb{1}\{z_{t,i} > 0 \lor z_{t+1,i} > 0\}} $$

Añadiendo una penalización a la inestabilidad del soporte temporal $1 - \mathcal{J}(z_t, z_{t+1})$ (la Pérdida de Jaccard Temporal o TJ Loss), se asegura que el perfil de esparcidad permanezca estable en el tiempo durante interacciones físicas estáticas, reestructurándose bruscamente solo al entrar o salir de contacto.

## 4. Referencias
- [[2026_LpWM]]
- [[LpWM]]
- [[SIGReg]]

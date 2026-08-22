---
title: "Sketched-Isotropic-Gaussian Regularizer (SIGReg)"
tags: [math, regularization, loss, anti-collapse]
---

# Sketched-Isotropic-Gaussian Regularizer (SIGReg)

## Motivación
En las arquitecturas JEPA, predecir el futuro estado latente utilizando solo el error cuadrático medio (MSE) conduce al **colapso de la representación**, donde el encoder mapea todas las entradas a un vector constante (típicamente cero). Para evitar esto, Balestriero y LeCun (2025) introdujeron **SIGReg**, que fuerza a los embeddings a seguir una distribución Gaussiana isotrópica, garantizando la diversidad de características.

## Formulación Matemática

Sea $Z \in \mathbb{R}^{N \times B \times d}$ el tensor de embeddings latentes recolectados, donde $d$ es la dimensión del embedding. 

Evaluar la normalidad directamente en altas dimensiones es computacionalmente inestable y difícil. SIGReg utiliza el **Teorema de Cramér-Wold**, que establece que si todas las proyecciones marginales 1D de una distribución coinciden con una distribución objetivo, entonces la distribución conjunta completa también coincide.

### 1. Proyección Aleatoria
Se proyectan los embeddings $Z$ sobre $M$ direcciones aleatorias $u^{(m)}$ muestreadas uniformemente de la hiperesfera unidad $\mathbb{S}^{d-1}$:
$$ h^{(m)} \triangleq Z u^{(m)}, \quad u^{(m)} \in \mathbb{S}^{d-1} $$

### 2. Matching de Distribución 1D (Test Epps-Pulley)
Para cada proyección 1D, se evalúa la normalidad utilizando el estadístico de prueba univariado de **Epps-Pulley ($T$)**. La pérdida SIGReg es el promedio de este estadístico sobre las $M$ proyecciones:

$$ \text{SIGReg}(Z) \triangleq \frac{1}{M} \sum_{m=1}^{M} T(h^{(m)}) $$

Donde la prueba de Epps-Pulley se define mediante la función característica empírica:
$$ T^{(m)} = \int_{-\infty}^{\infty} w(t) \left| \phi_N(t; h^{(m)}) - \phi_0(t) \right|^2 dt $$

- $\phi_N$: Función característica empírica de las proyecciones.
- $\phi_0$: Función característica de la Gaussiana estándar $\mathcal{N}(0, 1)$.
- $w(t)$: Función de peso, ej. $w(t) = e^{-\frac{t^2}{2\lambda^2}}$.

Al converger en el límite asintótico para $M$, se obtiene la convergencia débil hacia la Gaussiana isotrópica:
$$ \text{SIGReg}(Z) \to 0 \iff \mathbb{P}_Z \to \mathcal{N}(0, I) $$

## Referencias
- [[2026_LeWorldModel]]: Usado como el único término de regularización anti-colapso.

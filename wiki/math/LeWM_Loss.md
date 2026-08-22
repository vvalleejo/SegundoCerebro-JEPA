---
title: "LeWorldModel Loss Objective"
tags: [math, loss, jepa, leworldmodel]
---

# LeWorldModel Loss Objective ($\mathcal{L}_{LeWM}$)

La función de pérdida total empleada para entrenar la arquitectura [[LeWorldModel]] es extremadamente minimalista, reduciendo la complejidad de otras implementaciones JEPA (que pueden llegar a tener hasta 6 hiperparámetros) a solo un hiperparámetro crítico.

## Formulación

$$ \mathcal{L}_{LeWM} \triangleq \mathcal{L}_{pred} + \lambda \text{SIGReg}(Z) $$

Donde:
1. **Prediction Loss ($\mathcal{L}_{pred}$)**: Es el término que incentiva al modelo a aprender dinámicas. Usa un Error Cuadrático Medio (MSE) clásico (Teacher-Forcing) entre el estado latente predicho $\hat{z}_{t+1}$ y el embedding real $z_{t+1}$ codificado en el tiempo $t+1$.
   $$ \mathcal{L}_{pred} \triangleq \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \quad \hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t) $$

2. **Regularization Loss ([[SIGReg]])**: Evita el colapso trivial donde el encoder colapsa y arroja salidas constantes.
   $$ \text{SIGReg}(Z) \triangleq \frac{1}{M} \sum_{m=1}^{M} T(h^{(m)}) $$

3. **$\lambda$**: El peso de la regularización. Debido a la formulación robusta de SIGReg, $\lambda$ es el *único* hiperparámetro que requiere ajuste fino durante el entrenamiento, y se puede optimizar en tiempo logarítmico $\mathcal{O}(\log n)$ usando búsqueda por bisección (un valor típico reportado en la literatura es $\lambda = 0.1$).

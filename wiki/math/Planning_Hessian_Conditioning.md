---
title: "Condicionamiento del Hessiano de planificación (dinámica lineal ε-straight)"
type: math
used_by: ["[[Temporal_Straightening]]"]
tags: [math, theorem, planning, conditioning, controllability-gramian, gradient-based-planning]
---

# Condicionamiento del Hessiano de planificación

Teorema 4.4 / C.5 de [[wiki/papers/2026_Temporal_Straightening|Temporal Straightening for Latent Planning]]. Relaciona la "rectitud" de la dinámica latente con el **número de condición** del problema de planificación por gradiente, y por tanto con su velocidad de convergencia.

## 1. Setup (PDF §4; App. C.1)

- Acciones $\mathbf a=(a_0,\dots,a_{K-1})\in\mathbb R^{K\times d_a}$; $K$ = **horizonte de planificación** (canónico: $H$; en el resto del paper $K$ también es la historia del predictor).
- Estado inicial fijo $z_0\in\mathbb R^d$ y objetivo $z_g\in\mathbb R^d$.
- Coste terminal: $\mathcal L(\mathbf a)=\|z_K-z_g\|_2^2$, $z_K=\Phi(\mathbf a)$, con $\Phi$ el rollout (Eq. 8, 14).
- **Assumption 4.1 / C.1 (lineal):** $z_{t+1}=Az_t+Ba_t$, $A\in\mathbb R^{d\times d}$, $B\in\mathbb R^{d\times d_a}$ (Eq. 9, 15).
- **Def. 4.2 / C.3 ($\varepsilon$-straight):** $\|A-I\|_2\le\varepsilon$ (Eq. 10). Con $\varepsilon\to0$ la dinámica tiende a $z_{t+1}=z_t+Ba_t$: trayectoria recta modificada solo por el control.
- **Def. C.2:** para $H\succeq0$ con núcleo no trivial, $\kappa_{\text{eff}}(H):=\sigma_{\max}(H)/\sigma^+_{\min}(H)$, con $\sigma^+_{\min}$ el menor valor singular **no nulo**. $\kappa(A):=\sigma_{\max}(A)/\sigma_{\min}(A)$.

## 2. Hessiano y Gramiano (Lemma C.4)

Desenrollando: $z_K=A^Kz_0+\sum_{t=0}^{K-1}A^{K-1-t}Ba_t$ (Eq. 16), afín en $\mathbf a$. Con el Jacobiano del rollout

$$J_\Phi=\frac{\partial z_K}{\partial\mathbf a}=\big[A^{K-1}B,\ A^{K-2}B,\ \dots,\ B\big]\in\mathbb R^{d\times Kd_a}\quad\text{(Eq. 17)}$$

$$H:=\nabla^2_{\mathbf a}\mathcal L(\mathbf a)=2J_\Phi^\top J_\Phi\succeq0,\qquad \mathcal W_K:=J_\Phi J_\Phi^\top=\sum_{k=0}^{K-1}A^kBB^\top(A^\top)^k\in\mathbb R^{d\times d}\quad\text{(Eq. 18–19)}$$

$\mathcal W_K$ es el **Gramiano de controlabilidad de horizonte finito**. Como los autovalores no nulos de $J^\top J$ y $JJ^\top$ coinciden, $\kappa_{\text{eff}}(H)=\kappa(\mathcal W_K)$ (Eq. 20).

> [!important] $H$ es singular
> $H\in\mathbb R^{Kd_a\times Kd_a}$ tiene rango $\le d$. Con $d_a=d$ y $K>1$ tiene núcleo no trivial: hay muchas secuencias de acciones que alcanzan el mismo $z_K$. De ahí el uso de $\kappa_{\text{eff}}$.

## 3. Teorema (Thm 4.4 / C.5)

**Hipótesis:** Assumption 4.1, $d_a=d$ y $B$ **invertible**. Entonces

$$\kappa_{\text{eff}}(H)=\kappa(\mathcal W_K)\le\kappa(B)^2\,\frac{\sum_{k=0}^{K-1}\sigma_{\max}(A)^{2k}}{\sum_{k=0}^{K-1}\sigma_{\min}(A)^{2k}}\le\kappa(B)^2\,\kappa(A)^{2(K-1)}\quad\text{(Eq. 11, 21)}$$

Si además $\varepsilon=\|A-I\|_2<1$:

$$\kappa_{\text{eff}}(H)\le\kappa(B)^2\Big(\frac{1+\varepsilon}{1-\varepsilon}\Big)^{2(K-1)}\quad\text{(Eq. 12, 22)}$$

y para $\varepsilon\le\tfrac12$, $\kappa_{\text{eff}}(H)\le\kappa(B)^2e^{6\varepsilon K}$.

**Esbozo de la prueba (App. C.2):**
- Cota superior: $x^\top\mathcal W_Kx=\sum_k\|B^\top(A^\top)^kx\|_2^2\le\sigma_{\max}(B)^2\sum_k\sigma_{\max}(A)^{2k}$.
- Cota inferior: $\|B^\top u\|_2\ge\sigma_{\min}(B)\|u\|_2$ y $\sigma_{\min}(A^k)\ge\sigma_{\min}(A)^k$.
- El cociente de sumas está acotado por el máximo cociente término a término.
- Weyl: $\sigma_{\max}(A)\le1+\varepsilon$, $\sigma_{\min}(A)\ge1-\varepsilon$. Finalmente $\ln\frac{1+\varepsilon}{1-\varepsilon}\le3\varepsilon$ para $\varepsilon\le\frac12$.

**Consecuencia:** como $\mathcal L$ es cuadrática convexa, GD converge linealmente con una tasa controlada por $\kappa_{\text{eff}}(H)$. Una dinámica más recta ($\varepsilon$ pequeño) da un $\kappa$ que crece despacio con el horizonte.

## 4. Caso $d_a<d$ (Remark 4.5 / C.6)

$B$ no es invertible y $\mathcal W_K$ puede ser singular. Los enunciados valen en el subespacio controlable $\mathcal S_K=\mathrm{range}(\mathcal W_K)$, sustituyendo $\lambda_{\min}$ por $\lambda^+_{\min}$, pero **hacen falta hipótesis de controlabilidad adicionales** para acotar inferiormente $\sigma^+_{\min}(\mathcal W_K)$. No se dan.

> [!warning] Distancia entre teoría y experimentos
> - Todos los experimentos tienen $d_a=2\ll d$, así que el teorema principal no se aplica literalmente.
> - El predictor real es un ViT no lineal. Los autores dejan como trabajo futuro controlar productos de Jacobianos dependientes del estado.
> - El enlace entre la pérdida entrenada (coseno) y $\varepsilon$ pasa por [[Temporal_Straightening_Loss]] §3, que es **direccional** y supone velocidad de norma constante.
> - La cota exponencial es débil ($\varepsilon=0.5$, $K=5$ ⇒ factor $e^{15}$). Su lectura útil es cualitativa: $\varepsilon\to0$ ⇒ $\kappa_{\text{eff}}\to\kappa(B)^2$, independiente de $K$.
> - La evidencia empírica en el caso no lineal es el paisaje de pérdida de la Fig. 4 (una sola muestra de PushT).

## Relación con la bóveda

- Complementa el análisis de error acumulado de [[Latent_Dynamics_Consistency]] (SG-JEPA): allí $\hat A^{h-1-j}$ amplifica el error de predicción; aquí $A^k$ amplifica el mal condicionamiento de la planificación. En ambos casos, que el espectro de $A$ se mantenga cerca de 1 es lo que controla el crecimiento con el horizonte.
- Motiva la sección de planificación por gradiente de [[JEPA-master-note]] §7.

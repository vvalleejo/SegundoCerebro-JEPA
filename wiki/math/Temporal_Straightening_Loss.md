---
title: "Temporal Straightening Loss (regularizador de curvatura latente)"
type: math
used_by: ["[[Temporal_Straightening]]", "[[Semantic_Tube]]"]
tags: [math, regularizer, curvature, temporal-straightening, cosine-similarity]
---

# Temporal Straightening Loss

Regularizador geométrico que **penaliza la curvatura local** de las trayectorias latentes. Lo introduce en world models [[wiki/papers/2026_Temporal_Straightening|Temporal Straightening for Latent Planning]] (ICML 2026). **No es un mecanismo anti-colapso**: se suma a la pérdida predictiva, que usa stop-gradient.

## 1. Definición (PDF §3.2–3.3)

Sea $o_t\in\mathbb R^{n_o}$ la observación y $z_t=\mathcal E^s_\phi(o_t)\in\mathbb R^{d}$ su latente, con $\mathcal E^s_\phi$ el sensory encoder de parámetros $\phi$. Para tres latentes consecutivos:

$$v_t=z_{t+1}-z_t\in\mathbb R^d,\qquad v_{t+1}=z_{t+2}-z_{t+1}\in\mathbb R^d \quad\text{(Eq. 3)}$$

$$\mathcal C=\frac{v_t^\top v_{t+1}}{\|v_t\|_2\,\|v_{t+1}\|_2}\in[-1,1],\qquad \mathcal L_{\text{curv}}=1-\mathcal C\in[0,2] \quad\text{(Eq. 4, 6)}$$

Objetivo total, **forma aditiva**:

$$\mathcal L_{\text{total}}=\mathcal L_{\text{pred}}+\lambda\,\mathcal L_{\text{curv}},\qquad \mathcal L_{\text{pred}}=\|\hat z_{t+1}-\mathrm{sg}(z_{t+1})\|_2^2,\quad\lambda\ge0 \quad\text{(Eq. 5, 7)}$$

- $\hat z_{t+1}\in\mathbb R^d$: predicción de $f_\theta$; $\mathrm{sg}$: stop-gradient.
- $\lambda=0.1$ en features espaciales; $\{0.1, 0.01, 0.001\}$ en globales (Tab. 1).
- $\mathcal L_{\text{curv}}$ se calcula sobre latentes **codificados** (no predichos) y, por tanto, su gradiente llega **solo al encoder** (y a la cabeza de agregación), no al predictor.

> [!important] Invariancia de escala
> $\mathcal C$ es invariante a la norma de $v_t$: la pérdida penaliza el **cambio de dirección**, no la velocidad. Por eso no induce colapso por sí misma, a diferencia de la penalización de suavidad $\mathcal L_{\text{smooth}}=\mathbb E_t\|z_{t+1}-z_t\|_2^2$, cuyo mínimo trivial es un embedding constante (App. B.5). A cambio, $\mathcal C$ no está definido si $v_t=0$. El paper no dice cómo trata ese caso (presumiblemente con un $\epsilon$ numérico).

## 2. Variantes para features espaciales (App. B.6)

Con $z^v_t\in\mathbb R^{m_v\times d_v}$ ($m_v>1$ tokens, $d_v$ canales), $v_t=z^v_{t+1}-z^v_t$, $v_{t,i}\in\mathbb R^{d_v}$ el parche $i$ y $\cos(u,w)=\frac{u^\top w}{\|u\|_2\|w\|_2}$:

| Variante | $\mathcal C_t$ |
|---|---|
| patch | $\frac1{m_v}\sum_{i=1}^{m_v}\cos(v_{t,i},v_{t+1,i})$ |
| mean | $\cos(\bar v_t,\bar v_{t+1})$, $\bar v_t=\frac1{m_v}\sum_i v_{t,i}$ |
| flatten | $\cos(\mathrm{vec}(v_t),\mathrm{vec}(v_{t+1}))$, $\mathrm{vec}:\mathbb R^{m_v\times d_v}\to\mathbb R^{m_vd_v}$ |
| **agg** (principal) | $\cos(h_\phi(v_t),h_\phi(v_{t+1}))$, $h_\phi:\mathbb R^{m_v\times d_v}\to\mathbb R^{d_h}$, MLP con $d_h=128$ |

$\lambda=0.1$ para agg y $0.01$ para las demás (App. B.6). *agg* es la mejor en 3 de 4 entornos; en PointMaze-Medium gana *flatten* (Fig. 14).

> [!question] No verificado
> La notación del paper aplica $h_\phi$ a $v_t$ (diferencias), mientras que la Fig. 13 aplica la cabeza de agregación a $z_t$ y el coseno a $g_t=h_\phi(z_t)$. Si $h_\phi$ es un MLP no lineal, $h_\phi(z_{t+1}-z_t)\neq h_\phi(z_{t+1})-h_\phi(z_t)$. El PDF no aclara cuál se implementa.

## 3. Justificación teórica: coseno ⇒ $(A-I)$ pequeño (App. C.3)

**Hipótesis (Assumption C.7):** dinámica lineal $z_{t+1}=Az_t+Ba_t$ ($A\in\mathbb R^{d\times d}$, $B\in\mathbb R^{d\times d_a}$); velocidad de **norma constante** $\|v_t\|_2=c>0$ para todo $t$; acciones suaves $\Delta_a:=\max_t\|a_{t+1}-a_t\|_2<\infty$.

**Prop. C.9.** Con $\hat v_t=v_t/\|v_t\|_2$ y $\mathcal C_t=\cos(v_t,v_{t+1})$, para $t=0,\dots,K-2$:

$$\|(A-I)\hat v_t\|_2\le\sqrt{2(1-\mathcal C_t)}+\frac{\sigma_{\max}(B)\,\Delta_a}{c}\quad\text{(Eq. 23)}$$

y si $\bar{\mathcal C}:=\frac1{K-1}\sum_{t=0}^{K-2}\mathcal C_t\ge1-\eta$:

$$\frac1{K-1}\sum_{t=0}^{K-2}\|(A-I)\hat v_t\|_2\le\sqrt{2\eta}+\frac{\sigma_{\max}(B)\,\Delta_a}{c}\quad\text{(Eq. 24)}$$

**Esbozo:** $v_{t+1}-v_t=(A-I)v_t+B(a_{t+1}-a_t)$; desigualdad triangular; con normas iguales, $\|v_{t+1}-v_t\|_2^2=2c^2(1-\mathcal C_t)$; Jensen sobre $\sqrt{\cdot}$ (cóncava) para el promedio.

> [!warning] Alcance
> - La cota es **direccional**: controla $A-I$ solo en las direcciones visitadas $\{\hat v_t\}$. Para pasar a $\|A-I\|_2\le\varepsilon$ (lo que usa [[Planning_Hessian_Conditioning]]) hace falta una condición de **cobertura** (las direcciones visitadas generan $\mathbb R^d$), que el paper considera razonable pero no verifica (Remark C.10).
> - $\|v_t\|_2$ constante es una hipótesis fuerte. El término $\sigma_{\max}(B)\Delta_a/c$ no se anula aunque $\mathcal C_t=1$.

## 4. Relación con otras formulaciones de la bóveda

- [[Semantic_Tube]]: $\mathcal L_{\text{STP}}=1-\cos(h_t-h_r,h_r-h_s)$ con $s<r<t$ sobre estados ocultos de un LLM. Es **la misma forma funcional**, con posiciones no necesariamente consecutivas; allí se justifica con la "hipótesis geodésica" y aquí con el condicionamiento de la planificación.
- [[wiki/papers/2026_MotionJEPA|MotionJEPA]] usa como métrica (no como pérdida) la rectitud global $S(z)=\frac{\|z_T-z_0\|_2}{\sum_t\|z_{t+1}-z_t\|_2}$. $\mathcal L_{\text{curv}}$ es su análogo **local** (segundo orden) y diferenciable.
- Contraste: [[Semigroup_Rollout_Consistency]] actúa sobre la consistencia de la **composición** del predictor, no sobre la geometría de las trayectorias codificadas.
- Antecedentes citados (no ingeridos): Goroshin, Mathieu & LeCun (2015) *Learning to linearize under uncertainty*; Hénaff et al. (2019) *Perceptual straightening*.

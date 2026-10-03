---
title: "Teacher-Forcing + Rollout Loss con TBPTT (JEPA-WM / AD-E2E-JEPA)"
type: math
used_by: ["[[AD-E2E-JEPA]]", "[[JEPA-WM]]"]
tags: [math, loss, rollout, teacher-forcing, tbptt, world-models, sigreg]
---

# Teacher-Forcing + Rollout Loss (con TBPTT)

Objetivo multi-paso de [[wiki/papers/2026_AD-E2E-JEPA|AD-E2E-JEPA]] (Eq. 12, App. A.2). Adapta la implementación de rollout de JEPA-WM (Terver et al., 2025, no ingerido) a todo el horizonte futuro $F$. Combina tres términos: teacher forcing, consistencia de rollout autorregresivo con *truncated backpropagation through time* (TBPTT) y [[SIGReg]] sobre toda la ventana.

## 1. Notación

| Símbolo | Tipo / dimensión | Significado |
|---|---|---|
| $K$ | entero; $K+1=4$ | longitud de la ventana de contexto del predictor (historia) |
| $F$ | entero; $F=8$ | horizonte futuro (canónico: $H$) |
| $z_j$ | $\mathbb R^{H'\times W'\times D}=\mathbb R^{4\times8\times256}$ | embedding proyectado del frame $j$, $z_j=\mathrm{Proj}(\mathrm{Enc}(I_j))$ |
| $a_j$ | $\mathbb R^3$ | acción (pose relativa) entre $j$ y $j+1$ |
| $E_a$ | lineal | encoder de acción |
| $\mathrm{Pred}'$ | red | predictor: ventana de $K+1$ embeddings + acciones → ventana desplazada un paso |
| $\mathrm{sg}$ | operador | stop-gradient |
| $\mathrm{MSE}$ | escalar | error cuadrático medio sobre los elementos indicados, $\propto\|\cdot\|_2^2$ |

Cada ventana de predicción abarca $K+2$ frames consecutivos: $K+1$ de entrada y la predicción desplazada un paso (App. A.2).

## 2. Teacher forcing (App. A.2.1)

Para $k=0,\dots,F-1$ se usa el contexto **real** (sin rollout):

$$\hat z^{\text{TF}}_{t+k-K+1:t+k+1}=\mathrm{Pred}'\big(z_{t+k-K:t+k},\,E_a(a_{t+k-K:t+k})\big)\quad\text{(Eq. 15)}$$

- Primera ventana ($k=0$): se supervisa **toda** la ventana, es decir, $K+1$ predicciones:
$$\mathcal L^{\text{proj}}_{\text{pred}}[0]=\mathrm{MSE}\big(\hat z^{\text{TF}}_{t-K+1:t+1},\,\mathrm{sg}(z_{t-K+1:t+1})\big)\quad\text{(Eq. 16)}$$
- Ventanas siguientes ($k\ge1$): solo el **último** paso:
$$\mathcal L^{\text{proj}}_{\text{pred}}[k]=\mathrm{MSE}\big(\hat z^{\text{TF}}_{t+k+1},\,\mathrm{sg}(z_{t+k+1})\big),\quad k=1,\dots,F-1\quad\text{(Eq. 17)}$$
- Media ponderada por el número de predicciones supervisadas, $(K+1)+(F-1)=K+F$:
$$\mathcal L_{\text{TF}}=\frac{(K+1)\,\mathcal L^{\text{proj}}_{\text{pred}}[0]+\sum_{k=1}^{F-1}\mathcal L^{\text{proj}}_{\text{pred}}[k]}{K+F}\quad\text{(Eq. 18)}$$

## 3. Consistencia de rollout con TBPTT (App. A.2.2)

Un único rollout autorregresivo de $F$ pasos desde el contexto real, con una ventana deslizante de $K+1$ embeddings:

$$\tilde z^{\text{AR},(0)}_{t-K:t}=z_{t-K:t}\quad\text{(Eq. 19)}$$

$$\hat z^{\text{AR},(1)}_{t-K+1:t+1}=\mathrm{Pred}'\big(\tilde z^{\text{AR},(0)}_{t-K:t},E_a(a_{t-K:t})\big)\quad\text{(Eq. 20; reutiliza la predicción TF de } k=0\text{)}$$

$$\hat z^{\text{AR},(j)}_{t-K+j:t+j}=\mathrm{Pred}'\Big(\mathrm{sg}\big(\tilde z^{\text{AR},(j-1)}_{t-K+j-1:t+j-1}\big),E_a(a_{t-K+j-1:t+j-1})\Big),\quad j=2,\dots,F\quad\text{(Eq. 21)}$$

$$\tilde z^{\text{AR},(j)}_{t-K+j:t+j}=\Big[\tilde z^{\text{AR},(j-1)}_{t-K+j:t+j-1},\ \hat z^{\text{AR},(j)}_{t+j}\Big]\quad\text{(Eq. 22: se descarta el más antiguo y se añade la última predicción)}$$

$$\mathcal L_k=\mathrm{MSE}\big(\hat z^{\text{AR},(k)}_{t+k},\,\mathrm{sg}(z_{t+k})\big),\quad k=2,\dots,F\quad\text{(Eq. 23)}$$

El primer paso ya lo supervisa $\mathcal L_{\text{TF}}$ y por eso se excluye. Por el $\mathrm{sg}$ de la Eq. 21, cada $\mathcal L_k$ es diferenciable **solo a través de la llamada actual** al predictor (TBPTT de longitud 1).

## 4. Objetivo total (Eq. 12, App. A.2.3)

$$\mathcal L_{\text{multi}}=\frac{\mathcal L_{\text{TF}}+\mathcal L_2+\dots+\mathcal L_F}{F}+\lambda\,\mathcal L^{t-K:t+F}_{\text{SIGReg}}$$

Forma **aditiva**; SIGReg parche a parche sobre todos los frames de $t-K$ a $t+F$ (ver [[SIGReg]], fila AD-E2E-JEPA). $\lambda=0.09$ (Tab. 1).

> [!important] Qué hace y qué no hace el TBPTT
> Con $\mathrm{sg}$ sobre el contexto autorregresivo, el gradiente de $\mathcal L_k$ no llega a las predicciones anteriores. La pérdida enseña al predictor a **funcionar bien con entradas generadas por él mismo** (corrige el *exposure bias*), pero no optimiza cómo un error en el paso $j$ se propaga al paso $j+1$. Es la diferencia con [[Semigroup_Rollout_Consistency]], que retropropaga por todo el rollout (sin stop-gradient) y pondera los pasos con un descuento.

> [!important] Pesos implícitos
> Al dividir por $F$, $\mathcal L_{\text{TF}}$ (que ya es una media sobre $K+F$ predicciones) pesa lo mismo que cada $\mathcal L_k$ individual. El rollout recibe por tanto $(F-1)/F=7/8$ del peso predictivo total. El PDF no ablaciona esta ponderación.

## 5. Efecto empírico (Tab. 2)

En el test completo de NAVSIM (12 146 escenas, 256 candidatas), el rollout mejora la FDE de 6.3 a 4.5 m (navtrain) y de 6.2 a 4.0 m (trainval), y el hit rate top-1 de 32.9 a 42.6 % y de 31.0 a 53.8 %. En las 100 escenas, el EPDMS baja con navtrain (76.6 → 70.4).

## Relacionados

- [[Semigroup_Rollout_Consistency]], [[Latent_Dynamics_Consistency]]: rollout sin TBPTT y la teoría de por qué el teacher forcing de un paso puede invertir el ranking de modelos.
- [[LeWM_Loss]]: un paso, teacher forcing puro.

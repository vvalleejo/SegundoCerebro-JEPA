---
title: "Arquitectura EB-JEPA: Energy-Based Joint-Embedding Predictive Architecture"
paper: "[[2026_EB-JEPA]]"
entity: "[[EB-JEPA]]"
type: "architecture"
tags: [architecture, jepa, energy-based-models, modular, world-models, regularization]
---

# Arquitectura EB-JEPA (Energy-Based JEPA)

**EB-JEPA** formaliza el marco de Joint-Embedding Predictive Architectures bajo la teoría de **Modelos Basados en Energía (EBMs)** de Yann LeCun, ofreciendo una implementación modular y unificada que abarca desde el aprendizaje de representaciones estáticas en imágenes hasta Modelos de Mundo en video y control condicionado por acciones con rollouts multietapa.

---

## 1. Diagrama del Framework Unificado

![Framework Modular EB-JEPA](img/2026_EB-JEPA_arch.png)

---

## 2. Las Tres Instanciaciones Unificadas

```
(a) Image JEPA (SSL):
    x_context ---> [ Encoder E ] ---> s_context 
                                         |
                                   [ Predictor P ] ---> \hat{s}_target  <-- Energy E(x,y) -->  s_target <--- [ Encoder E (mismos pesos) ] <--- x_target

(b) Video JEPA (World Model):
    v_{1:t}   ---> [ Video Enc E ] ---> s_{1:t} 
                                         |
                                   [ Rollout P ]  ---> \hat{s}_{t+1:t+H} <-- Energy -->  s_{t+1:t+H} <--- [ Video Enc E ] <--- v_{t+1:t+H}

(c) Action-Conditioned JEPA (Control):
    s_t, a_{t:t+H-1} -------------> [ AC-Predictor ] ---> \hat{s}_{t+H}   <-- Min Energy -->  s_{goal}
```

---

## 3. Formulación de Energía y Regularización

La función de energía entre una observación $x$ y una predicción $y$ se define como:

$$\mathcal{E}(x, y) = \| P(E(x)) - E(y) \|_2^2$$

Para evitar el colapso de energía constante ($\mathcal{E}(x,y) = 0, \forall x,y$), EB-JEPA implementa y compara dos mecanismos fundamentales de regularización no contrastiva:
1. **VICReg (Variance-Invariance-Covariance)**: Regularización sobre matrices de covarianza empíricas.
2. **SIGReg (Sketched-Isotropic-Gaussian)**: Proyecciones aleatorias hacia distribuciones gaussianas estándar.

> [!important] Corrección (arbitraje con el PDF)
> La librería **no usa EMA ni stop-gradient**: "focusing on their subclass using regularization-based collapse prevention … rather than stop-gradient techniques" (EB-JEPA p. 2). Los diagramas de arriba se han corregido para reflejarlo. Qué regularizador usa cada ejemplo:
> - **Image-JEPA:** VICReg y SIGReg, comparados sobre un projector. En CIFAR-10 con ResNet-18, SIGReg obtiene 91.02 % y VICReg 90.12 % (Tab. 1).
> - **Video-JEPA:** varianza y covarianza, es decir, VICReg (Fig. 3).
> - **AC-video-JEPA:** $\mathcal L_{\text{pred}}+\alpha\mathcal L_{\text{var}}+\beta\mathcal L_{\text{cov}}+\delta\mathcal L_{\text{sim}}+\omega\mathcal L_{\text{IDM}}$ (Eq. 13). Planifica con MPPI (97 ± 2 %) o CEM (96 ± 2 %). **Sin IDM el éxito cae a 1 ± 1 %** (Tab. 4).

---

## 4. Referencias Cruzadas
- **Paper**: [[2026_EB-JEPA]]
- **Entidad**: [[EB-JEPA]]
- **Matemáticas**: [[SIGReg]], [[LeWM_Loss]]

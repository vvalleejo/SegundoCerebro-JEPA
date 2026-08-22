---
title: "VJEPA Loss Objective"
tags: [math, loss, variational, jepa]
---

# VJEPA Loss Objective ($\mathcal{L}_{VJEPA}$)

El objetivo de entrenamiento de [[VJEPA]] reemplaza la pérdida determinista (MSE) utilizada por JEPA o [[LeWorldModel]] con un enfoque variacional basado en la maximización de la log-verosimilitud (log-likelihood) y la minimización de la divergencia Kullback-Leibler (KL).

## Formulación

La pérdida se define como la Negative Log-Likelihood regularizada:

$$ \mathcal{L}_{VJEPA} = \mathbb{E}_{x} \mathbb{E}_{Z_T \sim q_{\theta'}(\cdot \mid x_T)} \left[ - \log p_\phi(Z_T \mid Z_C, \xi_T) \right] + \beta \mathbb{E}_{x} \text{KL} \left( q_{\theta'}(Z_T \mid x_T) \parallel p(Z_T) \right) $$

### Componentes:
1. **Predictive Negative Log-Likelihood (NLL)**:
   - El primer término entrena el predictor estocástico $p_\phi$ para que coincida con la distribución objetivo generada por el Target Encoder $q_{\theta'}$.
   - Dado que el Target Encoder genera una distribución (ej. una Gaussiana con media y varianza), se *muestrea* $Z_T$ utilizando el truco de reparametrización.
2. **Regularización KL ($\beta$)**:
   - El segundo término evita el colapso de las representaciones penalizando la divergencia de la distribución inferida del objetivo respecto a un prior estático $p(Z_T)$ (usualmente una Gaussiana isotrópica $\mathcal{N}(0, I)$).

## Maximización de Información Mutua
Desde una perspectiva teórica de la información, minimizar el término NLL de $\mathcal{L}_{VJEPA}$ equivale a maximizar una cota inferior variacional (Barber-Agakov bound) de la **Información Mutua Predictiva** entre el contexto pasado y el futuro: $I(Z_t ; Z_{t+\Delta})$. 

A diferencia de los modelos generativos (Auto-regresivos o VAEs) que fuerzan la codificación del ruido para reducir el error de reconstrucción de píxeles, VJEPA actúa como un filtro *Predictive Information Bottleneck (PIB)*, maximizando la retención de la señal predecible y siendo invariante al ruido estocástico del entorno (nuisance invariance).

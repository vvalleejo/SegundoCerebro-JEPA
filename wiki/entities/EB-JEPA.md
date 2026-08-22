---
title: "EB-JEPA (Energy-Based JEPA)"
tags: [entity, architecture, jepa, ebm, library]
---

# EB-JEPA (Energy-Based JEPA)

## Descripción
**EB-JEPA** es una biblioteca de software y un marco conceptual que enmarca las arquitecturas JEPA tradicionales como **Energy-Based Models (EBMs)**. Fue publicada en 2026 por FAIR y NYU. 

## Arquitectura Modular
EB-JEPA divide el proceso de World Modeling en componentes modulares y reemplazables:
1. **Encoders**: Convierten imágenes/videos en representaciones latentes.
2. **Predictors**: Redes que proyectan estados actuales (y acciones) en estados futuros predichos minimizando la *Energía* (prediction error).
3. **Regularizers**: Módulos dedicados puramente a evitar el colapso de representación, como VICReg o [[SIGReg]].
4. **Planners**: Algoritmos como MPPI o CEM que buscan trayectorias de mínima energía en el espacio latente.

## Multistep Rollout Training
Una contribución empírica importante de EB-JEPA es el entrenamiento "Multistep". En lugar de entrenar al predictor para predecir solo el paso $t+1$ dado el paso $t$ (Single-step), se entrena para predecir múltiples pasos en el futuro de forma recursiva ($t+1, t+2, \dots, t+k$), propagando el gradiente a través de las llamadas al predictor. Esto alinea la fase de entrenamiento con la fase de inferencia auto-regresiva, reduciendo drásticamente la acumulación de errores a largo plazo (exposure bias).

## Enlaces Relacionados
- Paper: [[2026_EB-JEPA]]

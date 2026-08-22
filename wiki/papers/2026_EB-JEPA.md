---
title: "A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures (EB-JEPA)"
authors: [Basile Terver, Randall Balestriero, Megi Dervishi, David Fan, Quentin Garrido, Tushar Nagarajan, Koustuv Sinha, Wancong Zhang, Mike Rabbat, Yann LeCun, Amir Bar]
year: 2026
tags: [paper, eb-jepa, library, energy-based-models, code]
---

# EB-JEPA: Biblioteca para Modelos Basados en Energía

## Resumen Ejecutivo
Este paper presenta **EB-JEPA**, una biblioteca de código abierto (open-source library) diseñada para fines educativos y de investigación rápida. EB-JEPA formula las arquitecturas predictivas (JEPA) bajo el prisma riguroso de los **Energy-Based Models (EBM)** introducidos por Yann LeCun.

## Formulación Energy-Based
En un EBM, el objetivo es aprender una función escalar de energía $E(x, y)$ que mida la compatibilidad entre entradas $x$ y salidas $y$. En el contexto de EB-JEPA, la "energía" se define como el error de predicción en el espacio de representación latente.

La biblioteca proporciona tres implementaciones incrementales:
1. **Image-JEPA**: Para invarianza de vistas en imágenes estáticas.
2. **Video-JEPA**: Para predicción temporal, introduciendo "multistep rollout training" que mejora la coherencia a largo plazo frente al entrenamiento de un solo paso.
3. **Action-Conditioned Video-JEPA**: Un World Model completo donde la predicción del futuro se condiciona por acciones (basado empíricamente en la arquitectura de [[V-JEPA2]]).

## Prevención del Colapso y Planificación
- **Regularización**: El paper resalta la superioridad de [[SIGReg]] (con un solo hiperparámetro de ajuste) frente a VICReg (que requiere sintonizar varianza, covarianza, similitud temporal, etc.) para mantener un entrenamiento estable en una sola GPU.
- **Control**: Utiliza Model Predictive Path Integral (MPPI), un algoritmo de optimización poblacional similar al Cross-Entropy Method (CEM), pero que asigna pesos suaves a las trayectorias muestreadas usando una distribución de Boltzmann basada en su coste/energía.

## Utilidad para el Doctorado
EB-JEPA proporciona la implementación práctica de referencia (código) para experimentar con variaciones matemáticas de los World Models latentes en entornos locales sin requerir clusters masivos de computación.

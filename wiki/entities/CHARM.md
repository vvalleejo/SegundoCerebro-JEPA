---
title: "CHARM (Channel-Aware Representation Model)"
tags: [entity, architecture, jepa, time-series, multimodal]
---

# CHARM

## Descripción
**CHARM** (2026, C3 AI) es una arquitectura de series temporales multivariante que aplica los principios de predicción latente de [[JEPA]] junto con el condicionamiento multimodal (texto + series temporales). 

## Arquitectura
1. **Text Embedding**: Usa un modelo de lenguaje para vectorizar las descripciones semánticas de cada sensor (canal).
2. **Contextual TCN**: Una red convolucional temporal donde los filtros y las puertas de activación (gates) se generan dinámicamente a partir de los embeddings de texto.
3. **Contextual Attention**: Capas de atención que cruzan la dimensión temporal y de canal, aplicando "Description-aware inter-channel attention gating" para suprimir interacciones espurias basadas en la semántica del sensor.
4. **Predictor JEPA**: Un predictor que aprende a predecir la representación latente de las ventanas temporales enmascaradas (causal prediction y smoothing).

## Relevancia
CHARM extiende el ecosistema JEPA, históricamente dominado por imágenes y video (ej. [[V-JEPA2]], [[LeWorldModel]]), al dominio industrial y de sensores, demostrando que predecir representaciones abstractas evita sobreajustarse al ruido estocástico de las mediciones físicas, permitiendo transferencia *Zero-Shot* entre configuraciones de sensores distintas.

## Enlaces Relacionados
- Paper: [[2026_CHARM]]

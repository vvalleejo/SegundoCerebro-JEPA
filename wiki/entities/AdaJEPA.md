---
title: "AdaJEPA (Adaptive JEPA)"
tags: [entity, architecture, jepa, world-models, test-time-adaptation, mpc]
---

# AdaJEPA (Adaptive JEPA)

## Descripción
**AdaJEPA** (2026, NYU & AMI Labs) es un paradigma de aprendizaje y control para World Models de la familia JEPA que implementa **Test-Time Adaptation (TTA)** continuo dentro del bucle cerrado de Model Predictive Control (MPC).

## Principios de Diseño
1. **Plan–Act–Adapt–Replan Loop**: Reemplaza el uso tradicional de modelos de mundo congelados tras la fase de entrenamiento offline.
2. **Auto-supervisión Online**: La propia interacción con el entorno (tras ejecutar una acción y observar la siguiente imagen/estado) genera una señal de entrenamiento latente gratuita.
3. **Actualizaciones Graduales Ligeras**: Solo se actualiza un pequeño subconjunto de parámetros (como las capas finales del predictor o encoder) usando 1 solo paso de gradiente por iteración de MPC, garantizando latencia mínima (0.01s - 0.03s).

## Relevancia para el Doctorado
AdaJEPA demuestra cómo resolver la brecha sim-to-real o la falta de robustez ante *test-time distribution shifts* (cambios de iluminación, masa, fricción o geometría de objetos). Es una pieza clave para conectar World Models estáticos con sistemas robóticos verdaderamente adaptativos e interactivos.

## Enlaces Relacionados
- Paper: [[2026_AdaJEPA]]
- Arquitecturas de Control Relacionadas: [[V-JEPA2]], [[LeWorldModel]], [[MuSe]]

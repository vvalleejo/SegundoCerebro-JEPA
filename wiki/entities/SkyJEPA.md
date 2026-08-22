---
title: "SkyJEPA"
tags: [entity, architecture, jepa, world-models, quadrotor, robotics, mppi]
---

# SkyJEPA

## Descripción
**SkyJEPA** (2026, UC Berkeley / NYU / Brown) es una arquitectura de modelo de mundo basada en [[JEPA]] diseñada para el control cinemático y dinámico en tiempo real de vehículos aéreos (cuadricópteros/drones) a alta frecuencia ($>100\text{ Hz}$) en hardware embebido.

## Componentes Arquitectónicos
1. **JEPA Latent Dynamics Predictor**: Utiliza encoders TCN e inhibe el colapso mediante regularización [[SIGReg]]. Predice la evolución del estado en un espacio abstracto sin reconstrucción explícita de sensores.
2. **Physics-Inspired Prober (PI Prober)**: Un módulo diferenciable que traduce los latentes congelados a variables físicas de estado $(p, v, R, \omega)$ estimando únicamente correcciones residuales sobre un modelo integrador cinemático en el grupo de Lie $SO(3)$.
3. **Control MPPI Embebido**: Integración en C++ con TensorRT para optimización por muestreo estocástico sobre chips NVIDIA Jetson.

## Relevancia para el Doctorado
SkyJEPA conecta el aprendizaje de representaciones latentes libres de reconstrucción con el control físico estricto y la interpretabilidad en robótica ágil. Demuestra que se puede hacer transferencia *Zero-Shot* de simulación a entorno real (*sim-to-real*) si se entrena con aleatorización de dominio y se acopla con un prober de física diferenciable.

## Enlaces Relacionados
- Paper: [[2026_SkyJEPA]]
- Arquitecturas de Control y Robótica: [[LeWorldModel]], [[V-JEPA2]], [[AdaJEPA]], [[MuSe]]

---
title: "MuSe (Multisensory Continual Learning)"
tags: [entity, architecture, continual-learning, robotics, world-models]
---

# MuSe

## Descripción
**MuSe** (Stanford, 2026) es un framework y arquitectura diseñada para expandir políticas de World Models robóticos pre-entrenados con nuevas modalidades sensoriales (específicamente retroalimentación de fuerza/torque) mediante *Continual Learning*.

## Mecanismos Principales
MuSe permite que un agente robótico integre nuevos sensores *después* del pre-entrenamiento masivo sin sufrir olvido catastrófico. Lo logra mediante:
1. **Multi-sensory World Modeling**: Fuerza al predictor latente a generar proyecciones futuras no solo de los píxeles o el estado del robot, sino también de la *nueva modalidad*.
2. **Replay con Enmascaramiento Modal**: Al reutilizar datos antiguos que no tienen el sensor de fuerza, MuSe utiliza máscaras latentes (similar al enmascaramiento de [[JEPA]]) para indicar la ausencia del sensor, optimizando solo las pérdidas de las modalidades presentes.

## Conexión con JEPAs y World Models
Aunque no se denomina formalmente un JEPA, MuSe comparte el núcleo filosófico de los World Models predictivos aplicados a robótica (como [[V-JEPA2-AC]]). Su objetivo predictivo multisensorial ayuda a alinear el espacio de representación latente, de forma muy análoga a la predicción cruzada (Cross-Modal Prediction) introducida en [[MJEPA]] o el condicionamiento textual en [[CHARM]].

## Enlaces Relacionados
- Paper: [[2026_MuSe]]

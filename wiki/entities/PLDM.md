---
title: "PLDM (Planning with a Latent Dynamics Model)"
tags: [entity, architecture, jepa, world-models, offline-learning, mppi, vicreg]
---

# PLDM (Planning with a Latent Dynamics Model)

## Descripción
**PLDM** (2025, NeurIPS) es un método de modelo de mundo latente basado en [[JEPA]] desarrollado por investigadores de NYU, Brown y FAIR para aprender la dinámica del entorno a partir de datos offline sin recompensas (*reward-free offline data*).

## Componentes y Funcionamiento
1. **Modelado de Dinámica en Espacio Latente**: Mapea observaciones a vectores latentes mediante un encoder $h_\theta(s)$ y predice la evolución latente futura con un conjunto de predictores $f_\theta^k(z, a)$.
2. **Regularización Anticolapso**: Utiliza regularización de varianza y covarianza al estilo de **VICReg** combinada con modelado de dinámica inversa (IDM) para evitar el colapso sin necesidad de decodificadores generativos de imágenes.
3. **Planificación con Incertidumbre (MPPI)**: Durante el despliegue, planifica secuencias de acciones en tiempo real con MPPI optimizando un coste de distancia a la meta más una penalización por incertidumbre (varianza del ensamble de predictores).

## Ventajas Clave
- **Generalización OOD**: Capaz de navegar en mapas y geometrías de obstáculos nunca vistos durante el entrenamiento.
- **Reutilización Multitarea**: Permite cambiar la tarea del agente (ej. pasar de alcanzar una meta a huir de un perseguidor) cambiando únicamente la función de coste del planificador, sin reentrenar el modelo de mundo.

## Enlaces Relacionados
- Paper: [[2025_PLDM]]
- Arquitecturas de Modelo de Mundo: [[LeWorldModel]], [[V-JEPA2]], [[AdaJEPA]], [[SkyJEPA]]

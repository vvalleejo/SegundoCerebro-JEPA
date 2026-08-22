---
title: "AdaJEPA: An Adaptive Latent World Model"
authors: [Ying Wang, Oumayma Bounou, Yann LeCun, Mengye Ren (NYU & AMI Labs)]
year: 2026
tags: [paper, adajepa, jepa, world-models, test-time-adaptation, mpc, robotics]
---

# AdaJEPA: Modelo de Mundo Latente Adaptativo

## Resumen Ejecutivo
Este paper (2026) introduce **AdaJEPA** (Adaptive Joint-Embedding Predictive Architecture), una extensión crucial para los modelos de mundo latentes en robótica y control. Aborda el problema de que los modelos de mundo JEPA tradicionales permanecen **congelados** durante la inferencia (despliegue/test-time). Cuando se produce un cambio en la distribución del entorno (cambios visuales de iluminación/ruido, o cambios físicos de masa, fricción o dinámica), un modelo de mundo congelado genera predicciones erróneas, provocando el fallo del planificador (MPC).

## El Problema del Modelo Congelado en Test-Time
En Model Predictive Control (MPC), el robot planifica una secuencia de acciones en el espacio latente del modelo de mundo, ejecuta la primera acción y vuelve a planificar (replan). Si el entorno real cambia respecto al conjunto de entrenamiento (ej. un objeto de otra forma, diferente masa o fricción), los errores del modelo latente se acumulan a lo largo del horizonte de planificación, derivando en acciones ineficaces o fallos catastróficos.

## La Solución: Bucle Plan–Act–Adapt–Replan
AdaJEPA introduce **Adaptación a Tiempo de Prueba (Test-Time Adaptation - TTA)** en el bucle cerrado de MPC sin necesidad de demostraciones de expertos ni recompensas:

1. **Planificación y Ejecución**: El modelo actual de AdaJEPA planifica mediante MPC (usando GD o CEM) y ejecuta el primer bloque de acciones.
2. **Señal de Supervisión Auto-supervisada**: Al ejecutar la acción $a_t$, el robot observa la transición real $(o_t, a_t, o_{t+1})$. La nueva observación $o_{t+1}$ sirve como target latente auto-supervisado $z_{t+1} = \mathcal{E}(o_{t+1})$.
3. **Adaptación Online**: Con tan solo **un paso de gradiente** por cada paso de replanificación en el búfer de experiencia reciente $\mathcal{B}$, AdaJEPA actualiza un subconjunto ligero de parámetros (las últimas capas del encoder y del predictor).
4. **Replanificación Recalibrada**: El modelo adaptado se reutiliza de inmediato en el siguiente paso de MPC con proyecciones corregidas.

## Resultados
- **Rendimiento Robusto**: En entornos de manipulación y navegación (PushT, PushObj, PointMaze), AdaJEPA duplica la tasa de éxito del modelo congelado ante cambios de forma de objeto, fallos visuales (ruido, desenfoque) y cambios en la física (masa y amortiguación).
- **Latencia Mínima**: Añade solo entre 0.01s y 0.03s por paso de replanificación de MPC.
- **Eficiencia de Datos**: La adaptación online compensa grandes reducciones en los datos de pre-entrenamiento. Un modelo AdaJEPA entrenado con solo 1k trayectorias supera a un modelo congelado entrenado con 64k trayectorias.

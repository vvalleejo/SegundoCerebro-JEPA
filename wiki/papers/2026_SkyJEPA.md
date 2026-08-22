---
title: "SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors"
authors: [Pratyaksh Rao, Wancong Zhang, Randall Balestriero, Yann LeCun, Giuseppe Loianno (UC Berkeley, NYU, Brown)]
year: 2026
tags: [paper, skyjepa, jepa, world-models, quadrotor, robotics, mppi, sim2real]
---

# SkyJEPA: Control de Cuadricópteros con Modelos de Mundo Latentes

## Resumen Ejecutivo
Este paper (2026) presenta **SkyJEPA**, el primer modelo de mundo latente de tipo JEPA diseñado para el **control en tiempo real de vehículos aéreos no tripulados (drones/cuadricópteros)** a alta frecuencia en hardware embarcado. Resuelve la paradoja del control en espacio latente: los modelos JEPA evitan el colapso y el error acumulado predictivo al no reconstruir imágenes o estados crudos, pero los planificadores robóticos necesitan coordenadas físicas reales (posición, velocidad, orientación) para evaluar restricciones de seguridad y costes de control.

## El Problema del Control Aéreo con NNs
1. **Modelos Autoregresivos Clásicos**: Predecir el siguiente estado de forma recursiva en el espacio de estados provoca una acumulación rápida de errores (compounding error), haciendo que las proyecciones a largo plazo se desvíen drásticamente (drift) de la física real.
2. **Abstracción vs. Controlabilidad**: Las representaciones latentes puras de JEPA son abstractas, lo que imposibilita verificar límites de aceleración o límites geográficos de seguridad de un dron directamente sobre los vectores latentes.

## Arquitectura y Componentes Clave
SkyJEPA combina tres elementos fundamentales:

1. **Modelo de Dinámica Latente JEPA**: Usa encoders TCN para historiales de estados y acciones, junto con un predictor GRU. Se entrena con una pérdida predictiva latente multi-paso regularizada con **SIGReg** para garantizar anisotropía y prevenir el colapso sin usar pérdidas de reconstrucción.
2. **Physics-Inspired Prober (PI Prober)**: Un mecanismo de prueba diferenciable que toma las representaciones latentes congeladas y las mapea a estados físicos interpretables $(p, v, R, \omega)$. En lugar de aprender la física desde cero, el prober predice *correcciones residuales de aceleración* ($\Delta \dot{v}, K_t a_t$) sobre un integrador cinemático nominal diferenciable (usando el mapa exponencial $SO(3)$).
3. **Control MPPI en Tiempo Real**: Incorporado dentro de un planificador óptimo estocástico MPPI en C++ optimizado con TensorRT sobre NVIDIA Jetson Orin NX a bordo del dron, logrando inferencias en $< 10\text{ ms}$ ($> 100\text{ Hz}$).
4. **Data Pipeline y Métrica TDQ**: Generación automática de datos en simulación con procesos gaussianos (GP) y aleatorización de dominio (Domain Randomization) sobre masa, inercia, arrastre aerodinámico y constantes de motores. Introducen la métrica **TDQ (Trajectory Distribution Quality)** para cuantificar la riqueza y cobertura del dataset.

## Resultados
- **Transferencia Sim-to-Real Zero-Shot**: Entrenado 100% en simulación y desplegado de forma segura en experimentos exteriores reales sin ningún ajuste previo.
- **Robustez**: Mantiene un seguimiento preciso de trayectorias incluso bajo variaciones no nominales en despliegue real (cambio de hélices, transporte de cargas de 300g).
- **Menor Error Acumulado**: Demuestra un error de proyección a largo plazo sensiblemente menor y trayectorias latentes más "rectas" (temporal straightening) que los modelos autorregresivos tradicionales.

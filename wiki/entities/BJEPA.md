---
title: "Bayesian JEPA (BJEPA)"
tags: [entity, architecture, jepa, bayesian, product-of-experts]
---

# Bayesian JEPA (BJEPA)

## Descripción
**Bayesian JEPA (BJEPA)** es una extensión modular de [[VJEPA]] diseñada para la planificación en espacios latentes que permite incorporar restricciones y metas (goals) de manera explícita sin reentrenar la dinámica del modelo (Zero-Shot Transfer).

## Innovación: Factorización Bayesiana (Product of Experts)
Mientras que un modelo predictivo tradicional fuerza al mismo predictor a aprender la dinámica del mundo y cómo alcanzar una meta específica simultáneamente, BJEPA desacopla esto usando una factorización basada en el **Product of Experts (PoE)**.

Calcula la probabilidad posterior del estado futuro $Z_T$ como el producto de dos expertos condicionalmente independientes:

$$ p(Z_T \mid Z_C, \eta) \propto p_{like}(Z_T \mid Z_C) \cdot p_{prior}(Z_T \mid \eta) $$

### 1. Likelihood Expert (Dinámicas)
$p_{like}(Z_T \mid Z_C)$ representa el modelo de transición físico. Es entrenado con datos crudos para entender cómo evoluciona el entorno, independientemente de la tarea (Task-Agnostic Physics).

### 2. Prior Expert (Restricciones / Metas)
$p_{prior}(Z_T \mid \eta)$ es el experto prior. Inyecta información auxiliar $\eta$ (ej. evitar obstáculos, llegar a una posición). Actúa limitando el espacio de estados predicho únicamente a aquellas trayectorias que cumplen la restricción de la tarea.

## Inferencia y Control
Matemáticamente, BJEPA implementa una actualización de filtro de Kalman en el espacio latente. Durante la planificación, se intersecan el manifold de los futuros físicamente posibles con el manifold de los futuros que cumplen la tarea. Esto permite:
- **Zero-Shot Transfer**: Cambiar de tarea equivale simplemente a cambiar el $p_{prior}$, manteniendo intacto el modelo físico.
- **Evitar Catastrophic Forgetting**: Al no reentrenar los pesos de las dinámicas, el modelo no olvida cómo funciona el mundo al aprender nuevas tareas.

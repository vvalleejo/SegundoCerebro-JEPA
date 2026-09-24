---
title: "DINO-WM"
type: entity
status: external-baseline
tags: [entity, baseline, world-models]
---

# DINO-WM (baseline externo)

> [!note] Stub
> DINO-WM no está ingerido en `raw/`. Esta ficha recoge **solo** lo que dicen de él los papers de la bóveda, para que los enlaces no queden colgantes. No es una descripción del paper original.

**Qué es, según la bóveda:** un world model que predice sobre las **features de parche congeladas de DINOv2**, con un predictor **causal autorregresivo** (C-JEPA p. 6) y planificación CEM. Es el baseline "foundation-model-based" contra el que se comparan los world models JEPA entrenados end-to-end.

| Comparación | Dato | Fuente |
|---|---|---|
| Tiempo de planificación | 47 s frente a 0.98 s de LeWM ("48×") | [[wiki/papers/2026_LeWorldModel\|LeWM]], Fig. 3 |
| Tokens por observación | ~200× más que LeWM; 196×384 frente a los 6×128 de C-JEPA | LeWM Fig. 3; C-JEPA Tab. 3 |
| Push-T (éxito) | 92.0 (tabla de LeWM); 91.33 (tabla de C-JEPA) | LeWM Tab. 5; C-JEPA Tab. 3 |
| Planificación en C-JEPA | 5 763 s frente a 673 s de C-JEPA (>8×) | C-JEPA p. 7 |
| Control con física OOD | Arm Catcher 9.5 % frente a 23.3 % de SG-JEPA | [[wiki/papers/2026_Semigroup-JEPA\|SG-JEPA]] pp. 8–9 |

**Por qué importa:** es el punto de comparación para la tesis de que un encoder **entrenado junto al predictor** ([[LeWorldModel]], [[SG-JEPA]]) produce latentes mejores para dinámica que un encoder fundacional congelado. SG-JEPA atribuye su ganancia sobre todo al encoder (experimento de crossover).

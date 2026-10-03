---
title: "JEPA-WM"
type: entity
status: external-baseline
tags: [entity, baseline, world-models, jepa]
---

# JEPA-WM (baseline externo)

> [!note] Stub
> JEPA-WM (Terver, Yang, Ponce, Bardes, LeCun; *What drives success in physical planning with joint-embedding predictive world models?*, arXiv:2512.24497, 2025) no está ingerido en `raw/`. Esta ficha recoge **solo** lo que dicen de él los papers de la bóveda.

**Qué es, según la bóveda:** un estudio de ablación sobre las decisiones de diseño de los world models JEPA. Según [[wiki/papers/2026_AD-E2E-JEPA|AD-E2E-JEPA]] (§3.2.1), su mejor configuración es:
- encoder **DINOv3 ViT-L** congelado, preferido frente a DINOv2 y V-JEPA 2;
- predictor con condicionamiento **AdaLN** y **RoPE**;
- entrenamiento con rollout opcional (teacher forcing + rollout con TBPTT, ver [[Teacher_Forcing_Rollout_Loss]]);
- planificación con CEM en su paper original (AD-E2E-JEPA §3.3.1).

Como el target sale del encoder congelado, se entrena solo con MSE, sin término anti-colapso (AD-E2E-JEPA §3.2.1).

| Comparación | Dato | Fuente |
|---|---|---|
| NAVSIM, 100 escenas, 256 candidatas | EPDMS 74.2, FDE 4.0 m, hit top-1/5 45/75 % | AD-E2E-JEPA Tab. 2 |
| Tiempo de planificación por escena | 101.0 s (A100), frente a 0.8 s de AD-E2E-JEPA | AD-E2E-JEPA Tab. 2 |
| Entrenamiento (navtrain) | 4×A100, batch 64, 13 h | AD-E2E-JEPA Tab. 1 |

**Por qué importa:** es la base de [[AD-E2E-JEPA]], que le añade un proyector comprimido con SIGReg. Representa el extremo "preciso pero caro" del trade-off entre eficiencia y precisión de planificación. Ver también [[DINO-WM]].

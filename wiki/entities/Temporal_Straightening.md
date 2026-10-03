---
title: "Temporal Straightening (JEPA world model con latentes enderezados)"
type: entity
tags: [entity, jepa, world-models, temporal-straightening, curvature, gradient-based-planning]
---

# Temporal Straightening

**Qué es:** un método de NYU / Brown / Toronto (Wang, Bounou, Zhou, Balestriero, Rudner, LeCun, Ren; ICML 2026) para aprender world models JEPA cuyo espacio latente sea **fácil de planificar con gradiente**. No es una arquitectura nueva: es la plantilla de [[DINO-WM]] (predictor ViT causal sobre features de parche) con el espacio latente entrenable y un regularizador de curvatura ([[Temporal_Straightening_Loss]]).

## Ideas clave
1. **Straightening implícito:** la pérdida JEPA por sí sola reduce la curvatura de las trayectorias latentes; el término explícito $\lambda(1-\cos(v_t,v_{t+1}))$ lo refuerza (PDF §5.2).
2. **Geometría ⇒ planificación:** con trayectorias rectas, la distancia euclídea latente se aproxima a la geodésica (Fig. 6), y en el caso lineal el Hessiano de planificación está mejor condicionado ([[Planning_Hessian_Conditioning]]).
3. **GD en vez de CEM:** con straightening, GD se acerca al éxito de CEM a ~1/10 del tiempo (App. B.3).
4. **Anti-colapso mínimo:** stop-gradient sin EMA.

## Resultado principal
Frente a DINO-WM con el mismo planificador GD: +20–60 puntos open-loop y +20–30 MPC en Wall, PointMaze (UMaze, Medium) y PushT (PDF §1; Tab. 1).

## Enlaces
- Paper: [[wiki/papers/2026_Temporal_Straightening|2026_Temporal_Straightening]]
- Arquitectura: [[wiki/architecture/2026_Temporal_Straightening_Arch|2026_Temporal_Straightening_Arch]]
- Matemáticas: [[Temporal_Straightening_Loss]], [[Planning_Hessian_Conditioning]]
- Relacionados: [[Semantic_Tube]] (misma pérdida, en LLMs), [[MotionJEPA]] (métrica de rectitud), [[AdaJEPA]] (mismo lab, mismo planificador GD), [[DINO-WM]] (baseline), [[AD-E2E-JEPA]] (retoma el proyector convolucional con stride $2	imes2$ y SIGReg en lugar de curvatura), [[JEPA]]

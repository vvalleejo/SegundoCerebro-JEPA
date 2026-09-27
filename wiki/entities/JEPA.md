---
title: "JEPA (Joint-Embedding Predictive Architecture)"
type: entity
tags: [entity, jepa, hub]
aliases: [Joint-Embedding Predictive Architecture]
---

# JEPA: Joint-Embedding Predictive Architecture

Nota **hub** de la familia. La explicación completa, con notación, derivaciones, colapso, world models, planificación y código, está en [[JEPA-master-note]].

**Definición mínima.** Una JEPA predice la **representación** de una parte de la señal ($s_y=E_{\bar\theta}(y)$) a partir de la representación de otra parte ($s_x=E_\theta(x)$), condicionada por $z$ (posiciones, acción o información estructural). No reconstruye la señal en el espacio de entrada. El reto central es evitar el **colapso representacional**.

## Variantes en la bóveda por mecanismo anti-colapso

| Mecanismo | Modelos |
|---|---|
| EMA + stop-gradient | [[I-JEPA]], [[V-JEPA2]], [[V-JEPA2.1]], [[TC-JEPA]], [[MJEPA]], [[CHARM]], [[HP-JEPA]], [[Music-JEPA]], [[VJEPA]] (+ KL) |
| Stop-gradient sin EMA | [[AdaJEPA]], [[Temporal_Straightening]] (+ regularizador de curvatura) |
| Encoder congelado | [[C-JEPA]] |
| VICReg (+ IDM) | [[PLDM]], [[EB-JEPA]] |
| SIGReg | [[LeJEPA]], [[LeVJEPA]], [[LeWorldModel]], [[SG-JEPA]], [[SkyJEPA]], [[MotionJEPA]] (vía [[DISReg]]) |
| RDMReg (sparse) | [[Rectified_LpJEPA]], [[LpWM]] |
| Sin predictor ni regularizador (predictor identidad) | [[Semantic_Tube]] |

**Con o sin predictor.** [[LeJEPA]], [[LeVJEPA]] y [[Rectified_LpJEPA]] **no tienen red predictora**: son objetivos de invarianza multi-vista más un regularizador distribucional. Pertenecen a la familia JEPA por nombre y por linaje, pero su tarea predictiva es trivial (predecir el centroide de las vistas globales).

**Fuera de la familia (contraste).** [[GeniWorld]] es generativo (flow matching). [[MuSe]] es una política generativa: la palabra "JEPA" no aparece en su PDF.

Teoría: [[Linear_Identifiability]], [[Isotropic_Gaussian_Optimality]], [[Latent_Dynamics_Consistency]].

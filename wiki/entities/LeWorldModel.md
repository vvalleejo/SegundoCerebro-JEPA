---
title: "LeWorldModel (LeWM)"
tags: [entity, architecture, jepa, world-models]
---

# LeWorldModel (LeWM)

## Descripción
**LeWorldModel (LeWM)** es una arquitectura introducida en 2026 por Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun y Randall Balestriero. Es la primera arquitectura tipo **[[JEPA]]** que logra entrenarse de forma completamente estable, end-to-end, desde píxeles en bruto, sin necesidad de heurísticas complejas (como las utilizadas en [[I-JEPA]] o [[V-JEPA]]).

## Arquitectura
Consiste en dos módulos principales entrenados conjuntamente:
- **Encoder ($enc_\theta$)**: Un Vision Transformer (ViT) que recibe observaciones crudas $o_t$ y produce un estado latente $z_t$. La proyección final ocurre utilizando un MLP de una capa tras el token `[CLS]`.
- **Predictor ($pred_\phi$)**: Un Transformer auto-regresivo condicionado a acciones a través de *Adaptive Layer Normalization (AdaLN)*. Recibe un historial de estados latentes y predice el embedding del siguiente estado $\hat{z}_{t+1}$ dada una acción $a_t$.

## Innovación Principal: Eliminación de Heurísticas
A diferencia de trabajos previos de JEPA, LeWM **no** requiere:
- Exponential Moving Averages (EMA) del encoder objetivo.
- Operaciones de Stop-Gradient (SG).
- Encoders pre-entrenados congelados (como DINOv2).
- Múltiples términos de pérdida de covarianza y varianza cruzada (como en VICReg/PLDM).

Logra esto reemplazando todo por un único término de regularización llamado **[[SIGReg]]**, que hace el balance de los embeddings forzándolos hacia una distribución Gaussiana isotrópica. Esto simplifica el tuneo de hiperparámetros a un solo escalar, $\lambda$.

## Enlaces Relacionados
- Paper: [[2026_LeWorldModel]]
- Función de pérdida principal: [[LeWM_Loss]]

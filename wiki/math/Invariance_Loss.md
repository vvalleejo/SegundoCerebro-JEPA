---
title: Invariance Loss
tags: [math, loss, jepa]
---

# Invariance Loss

La pérdida de invariancia (Invariance Loss) asegura que las representaciones (embeddings) generadas a partir de vistas locales estén emparejadas predictivamente con las vistas globales de la misma muestra. 

En arquitecturas como [[LeVJEPA]], se formaliza como el error cuadrático medio (MSE) entre el embedding de la vista global $z_0$ y cada una de las vistas locales $z_v$:

$$
\mathcal{L}_{inv} = \frac{1}{V + 1} \sum_{v=0}^V \|z_0 - z_v\|_2^2
$$

**Variables**:
- $V$: Número de vistas locales extraídas de un clip. En total hay $V + 1$ vistas considerando la vista global (indexada con $v=0$).
- $z_0$: Embedding latente del token `[cls]` de la vista global $x_0$, que cubre la extensión espacial más grande. Actúa como el target de predicción por construcción de las vistas.
- $z_v$: Embeddings latentes de las vistas locales recortadas espacialmente y aumentadas fotométricamente.

**Importancia**:
Dado que la red propaga los gradientes a través de $z_0$ y $z_v$ simultáneamente (sin un *stop-gradient* ni una *target network* congelada como en *V-JEPA*), optimizar esta función de pérdida de manera aislada conduciría de forma trivial a un **colapso representacional**, donde el modelo asigna una constante a todo input. Para contrarrestar esto de forma teórica en lugar de usar heurísticas, se combina con un regularizador en el espacio latente como [[SIGReg]].

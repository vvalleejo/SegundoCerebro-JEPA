---
title: LeVJEPA
tags: [architecture, entity, video, jepa]
---

# LeVJEPA

LeVJEPA es una arquitectura introducida en el paper [[2026_LeVJEPA]] que adapta el objetivo matemático de [[LeJEPA]] al dominio del video. 

## Diferencias Clave con V-JEPA
A diferencia de arquitecturas previas como **V-JEPA**, LeVJEPA descarta la asimetría arquitectónica clásica para evitar el colapso (como un *target encoder* con *exponential moving average*, un predictor y el *stop-gradient*). 

En su lugar, entrena **un único encoder** que procesa vistas globales y locales recortadas agresivamente, regularizado estadísticamente usando [[SIGReg]].

## Características
1. **Block-Causal Attention**: Utiliza atención bidireccional a nivel intra-frame y atención causal temporal inter-frame. Al no haber red predictora, la codificación respeta la causalidad de los eventos.
2. **Alta Eficiencia**: Utiliza una estrategia radical de *uniform random token dropping* (hasta el 95%) en vez de un *structured tube masking*. Esto no solo ahorra memoria y cómputo exponencialmente, sino que sirve como regularización estocástica.

Ver la arquitectura detallada en [[2026_LeVJEPA_Arch]].

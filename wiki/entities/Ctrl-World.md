---
title: "Ctrl-World"
type: entity
status: external-baseline
tags: [entity, baseline, world-models, generative]
---

# Ctrl-World (baseline externo)

> [!note] Stub
> Ctrl-World no está ingerido en `raw/`. Esta ficha recoge solo lo que dice de él [[GeniWorld]].

**Qué es, según la bóveda:** un world model **generativo** (vídeo) condicionado por acciones **numéricas**. Es el baseline directo de GeniWorld, que en su lugar condiciona con acciones *visuales* renderizadas desde el URDF.

| Métrica (RoboTwin, Clean-to-Random) | Ctrl-World | Fuente |
|---|---|---|
| LPIPS / PSNR / SSIM | 0.285 / 20.41 / 0.791 | [[wiki/papers/2026_GeniWorld\|GeniWorld]], Tab. I |
| FID / FVD / EWMScore | 21.66 / 35.85 / 51.47 | ídem |

Según GeniWorld, con distractores visuales Ctrl-World deja de correlacionar con el éxito real de las políticas, mientras que GeniWorld mantiene la correlación.

**Relación con JEPA:** ninguna directa. Sirve de contraste entre los world models que generan píxeles y los que predicen en latente ([[JEPA]]).

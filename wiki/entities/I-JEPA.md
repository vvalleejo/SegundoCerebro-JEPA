---
title: "I-JEPA (Image JEPA)"
tags: [entity, architecture, jepa, vision]
---

# I-JEPA (Image-based JEPA)

## Descripción
**I-JEPA** (2023) es la implementación fundacional de la arquitectura Joint-Embedding Predictive Architecture propuesta por Yann LeCun, aplicada al dominio de imágenes. Sirve como el ancestro directo de todos los modelos de mundo JEPA posteriores (V-JEPA, C-JEPA, etc.).

## Características Clave
A diferencia de los enfoques contrastivos, I-JEPA prescinde totalmente de transformaciones de datos hechas a mano (data augmentations). Aprende representaciones forzando al modelo a predecir las características semánticas de parches de imagen ocultos, basándose únicamente en el contexto visible de la misma imagen.

- **Estrategia de Enmascaramiento (Multi-block Masking)**: La innovación arquitectónica clave para que I-JEPA aprenda semántica de alto nivel es cómo selecciona los parches. En lugar de enmascarar parches aleatorios dispersos (como en MAE), I-JEPA enmascara **bloques contiguos semánticamente grandes** (target blocks) y usa un bloque de contexto espacialmente distribuido.
- **Eficiencia**: Al no tener que predecir píxeles, el predictor de I-JEPA es mucho más pequeño y rápido que un decoder generativo.

## Enlaces Relacionados
- Paper original: [[2023_I-JEPA]]
- Arquitecturas derivadas para video: [[V-JEPA2]], [[V-JEPA2.1]]

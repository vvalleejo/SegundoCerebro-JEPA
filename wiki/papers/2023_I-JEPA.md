---
title: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)"
authors: [Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas (Meta AI, McGill, NYU)]
year: 2023
tags: [paper, jepa, i-jepa, images, self-supervised]
---

# I-JEPA: Aprendizaje Auto-Supervisado desde Imágenes

## Resumen Ejecutivo
Este paper (2023) presenta **I-JEPA** (Image-based Joint-Embedding Predictive Architecture), la primera instanciación práctica y a gran escala de la visión de Yann LeCun sobre arquitecturas predictivas (JEPA) aplicada a imágenes estáticas. I-JEPA demuestra que es posible aprender representaciones visuales altamente semánticas *sin* usar "data augmentations" manuales (como recortes, cambios de color, etc.) y *sin* reconstruir píxeles.

## El Problema de Enfoques Previos
1. **Métodos Invariantes (Contrastivos)**: Modelos como DINO o SimCLR requieren aumentos de datos creados manualmente para generar "vistas" de la misma imagen. Esto introduce sesgos fuertes que pueden no generalizar bien a otras tareas (ej. segmentación vs clasificación) o modalidades.
2. **Métodos Generativos (MAE)**: Los Masked Autoencoders (MAE) enmascaran parches y obligan al modelo a reconstruirlos a nivel de píxel. Esto gasta capacidad de cómputo modelando detalles irrelevantes de alta frecuencia (ruido, texturas) en lugar de semántica de alto nivel.

## La Solución I-JEPA
I-JEPA propone predecir información faltante en un **espacio de representación abstracto**.
1. **Contexto**: Se toma un bloque grande de la imagen (context block) y se pasa por un "Context Encoder" (ViT).
2. **Target**: Se toman varios bloques objetivo de la misma imagen (target blocks) y se pasan por un "Target Encoder" (cuyos pesos son un EMA del Context Encoder).
3. **Predicción**: Un predictor ligero toma el contexto y la posición de los targets, y predice las representaciones de los targets. La pérdida es la distancia L2 en el espacio latente.

## Impacto
I-JEPA establece la base (2023) sobre la que se construirían después los modelos de video y control (V-JEPA, V-JEPA2, LeWorldModel). Demuestra empíricamente que:
- La predicción en el espacio latente aprende representaciones más semánticas que la reconstrucción de píxeles.
- Es altamente escalable y eficiente computacionalmente (más rápido que MAE o iBOT).

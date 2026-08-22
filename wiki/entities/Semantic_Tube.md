---
title: "Semantic Tube Prediction (STP)"
tags: [entity, architecture, jepa, llm, semantic-tube]
---

# Semantic Tube Prediction (STP)

## Descripción
**Semantic Tube Prediction (STP)** (2026) es una técnica de regularización para Large Language Models (LLMs) que implementa los principios de las arquitecturas [[JEPA]] (aislar la semántica predecible del ruido) en el dominio del procesamiento del lenguaje natural (NLP).

## Implementación Matemática
STP asume la "Hipótesis Geodésica": la trayectoria de los estados ocultos de un LLM al generar una frase coherente es localmente lineal. 
Por lo tanto, dados los estados ocultos $h_s, h_r, h_t$ para tres posiciones secuenciales $s < r < t$, el objetivo STP minimiza el ángulo (maximiza el coseno) entre los vectores direccionales:

$$ \mathcal{L}_{STP} = 1 - \cos(h_t - h_r, h_r - h_s) $$

Esta pérdida actúa como el objetivo de predicción de una arquitectura JEPA donde el predictor es una función de identidad (debido a la asunción de linealidad).

## Implicaciones para World Models
STP demuestra que los principios de JEPA (evitar modelar el ruido de alta frecuencia del entorno y centrarse en las características latentes predecibles) aplican no solo a la visión, el video y la robótica (como en [[LeWorldModel]], [[V-JEPA2]] o [[MuSe]]), sino también al *lenguaje como acciones en un espacio semántico abstracto*. Demuestra que los límites establecidos por las Chinchilla Scaling Laws pueden superarse modificando el objetivo de aprendizaje para mejorar la eficiencia semántica (Signal-to-Noise Ratio).

## Enlaces Relacionados
- Paper: [[2026_Semantic_Tube]]

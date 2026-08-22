---
title: "Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA"
authors: [Hai Huang, Yann LeCun, Randall Balestriero (Atlassian, NYU, Brown)]
year: 2026
tags: [paper, jepa, llm, nlp, semantic-tube]
---

# Semantic Tube Prediction (STP): JEPA para LLMs

## Resumen Ejecutivo
Este paper (2026) adapta la filosofía de los Joint-Embedding Predictive Architectures (JEPA) a los Modelos de Lenguaje Grande (LLMs). Cuestiona las leyes empíricas de escalado (Scaling Laws tipo Chinchilla), argumentando que describen un entrenamiento "típico" pero no el "óptimo". Propone una regularización llamada **Semantic Tube Prediction (STP)** que mejora drásticamente la eficiencia de datos (Signal-to-Noise Ratio), permitiendo que un LLM iguale su rendimiento usando **16 veces menos datos**.

## El Problema del Next Token Prediction (NTP)
El objetivo clásico de los LLMs es el *Next Token Prediction* ($\mathcal{L}_{NTP}$), que usa validación cruzada (Cross-Entropy). El problema es que el NTP confunde el "ruido estadístico superficial" con la "señal semántica global". Al inferir, los errores se acumulan y las trayectorias de estados ocultos colisionan, provocando el "colapso de modo" (mode collapse) o alucinaciones.

## La Hipótesis Geodésica y Semantic Tube
El paper plantea la **Hipótesis Geodésica**: las secuencias de tokens siguen trayectorias geodésicas (rutas de menor resistencia) en una variedad semántica suave, siendo localmente lineales.

**Semantic Tube Prediction (STP)** actúa como un regularizador estilo JEPA:
$$ \mathcal{L} = \mathcal{L}_{NTP} + \lambda \cdot \mathcal{L}_{STP} $$

$\mathcal{L}_{STP}$ confina las trayectorias de los estados ocultos a un "tubo" alrededor de esta línea geodésica. Penaliza cualquier desviación perpendicular (ruido) y recompensa el avance paralelo a la línea (señal). 

### Ventaja frente a otros LLM-JEPAs
Intentos previos de aplicar JEPA a LLMs requerían crear múltiples vistas del mismo texto artificialmente o usar predictores extra (añadiendo coste computacional). Con STP, como se asume linealidad local, el "predictor" se reduce a la función identidad. Calcula la similitud de coseno entre las diferencias de estados ocultos consecutivos en la secuencia, requiriendo un coste computacional extra casi nulo y sin "data augmentations".

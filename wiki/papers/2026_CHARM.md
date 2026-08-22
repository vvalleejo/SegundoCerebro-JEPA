---
title: "Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings"
authors: [Utsav Dutta, Gerardo Pastrana, Sina Khoshfetrat Pakazad, Henrik Ohlsson (C3 AI)]
year: 2026
tags: [paper, jepa, time-series, multimodal, charm]
---

# Giving Sensors a Voice: CHARM

## Resumen Ejecutivo
Este paper (2026) presenta **CHARM (Channel-Aware Representation Model)**, el primer modelo fundacional basado en JEPA diseñado para **series temporales multivariantes** (time-series). La innovación clave es que CHARM es un modelo *multimodal*: utiliza las descripciones textuales de los sensores (ej. "Sensor de temperatura del aceite") para condicionar la extracción de características temporales y la atención entre canales.

## El Problema en Series Temporales
Los modelos tradicionales de series temporales (incluyendo los foundation models recientes) tratan todos los canales de forma uniforme o los procesan de manera independiente. Si hay ruido o artefactos en un sensor, los enfoques basados en reconstrucción (autoencoders) desperdician capacidad modelando ese ruido.

## La Solución: Multimodal JEPA
CHARM adapta el objetivo de aprendizaje auto-supervisado de JEPA al dominio temporal:
1. **Representación Textual**: Se codifican las descripciones de texto de los sensores usando un LLM congelado.
2. **Contextual TCN**: Las convoluciones temporales usan *Kernel Gating* guiadas por los embeddings de texto, permitiendo que los filtros convolucionales se adapten a la semántica del sensor.
3. **Inter-Channel Attention**: La atención cruzada entre diferentes sensores se modula mediante las descripciones textuales, permitiendo aprender relaciones físicas (ej. cómo la válvula A afecta a la presión B).
4. **Pérdida JEPA**: En lugar de reconstruir los valores crudos de los sensores, CHARM predice los embeddings latentes de segmentos temporales futuros o enmascarados, haciéndolo robusto al ruido intrínseco de los sensores.

## Resultados
Al evaluar en tareas de previsión (forecasting), clasificación y detección de anomalías, CHARM (con un simple *linear probe* sobre el encoder congelado) supera a modelos de series temporales masivos. Demuestra que el uso del objetivo JEPA en lugar del error de reconstrucción es el factor que más contribuye al aumento del rendimiento en el modelado de series temporales.

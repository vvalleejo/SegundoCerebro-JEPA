---
title: "MJEPA: A Simple and Scalable Joint-Embedding Predictive Architecture for Audio-Visual Learning"
authors: [Revant Teotia, Adrien Bardes, Michael Rabbat, Sumit Chopra, Matthew Muckley, Nicolas Ballas (FAIR at Meta, NYU)]
year: 2026
tags: [paper, mjepa, multimodal, audio-visual, jepa]
---

# MJEPA: Aprendizaje Audio-Visual Conjunto

## Resumen Ejecutivo
Este paper (2026) presenta **MJEPA** (Multimodal Joint-Embedding Predictive Architecture), una extensión del framework JEPA diseñada para procesar conjuntamente flujos de audio y video. Demuestra que se puede utilizar un único encoder unificado para ambas modalidades sin recurrir a pérdidas contrastivas o reconstructivas.

## El Problema del Encoder Compartido
Estudios previos demostraron que al usar un único encoder (shared encoder) para procesar audio y video simultáneamente (en lugar de tener un encoder de visión y uno de audio separados), el rendimiento del modelo a menudo colapsaba y era inferior al de modelos entrenados en una sola modalidad. Esto se debe a que los pesos del encoder se "tiran" en dos direcciones opuestas intentando modelar dominios estadísticos diferentes.

## La Solución de MJEPA
MJEPA resuelve este conflicto introduciendo el concepto de **Cross-Modal Prediction** (Predicción Cruzada entre Modalidades) además del tradicional intra-modal prediction:
1. **Intra-modal**: Predecir audio enmascarado desde audio visible, y video enmascarado desde video visible.
2. **Cross-modal**: Predecir las representaciones del video usando *solo* el audio como contexto, y viceversa. 

Al forzar matemáticamente que la representación del audio pueda generar la representación latente del video, el modelo "alinea semánticamente" ambos dominios en el mismo espacio vectorial, logrando una transferencia positiva (sinergia) donde la información de una modalidad mejora a la otra.

## Resultados
Con un modelo escalado a 1 billón de parámetros (ViT-g), las representaciones congeladas de MJEPA superan a los modelos anteriores en tareas de clasificación como AudioSet-20K, Kinetics-400 y ESC-50, demostrando que JEPA es un framework altamente generalizable a configuraciones multimodales sin perder eficiencia.

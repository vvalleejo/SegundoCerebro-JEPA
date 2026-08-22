---
title: "Causal-JEPA: Learning World Models through Object-Level Latent Masking"
authors: [Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, Randall Balestriero]
year: 2026
tags: [paper, causal, jepa, object-centric, world-models]
---

# Causal-JEPA (C-JEPA): Modelos de Mundo Causales

## Resumen Ejecutivo
Este paper introduce **C-JEPA (Causal-JEPA)**, un modelo de mundo centrado en objetos (object-centric) que usa una estrategia de enmascaramiento a nivel de objeto para forzar al modelo a aprender interacciones causales. Es una desviación importante del enmascaramiento tradicional basado en parches (patch-based, como en [[V-JEPA2]]).

## El Problema del Enmascaramiento por Parches
Los modelos JEPA tradicionales enmascaran tubos de píxeles o parches. El problema es que el predictor puede resolver el objetivo reconstruyendo el futuro de un objeto basándose únicamente en la auto-dinámica del objeto (interpolación temporal trivial), ignorando cómo interactúa con el entorno o con otros objetos.

## La Solución: Object-Level Latent Masking
C-JEPA utiliza un encoder que extrae representaciones centradas en objetos (slots). Durante el entrenamiento, C-JEPA **enmascara toda la trayectoria histórica de un objeto** (dejando solo un "identity anchor" mínimo en $t_0$). 
Para predecir el estado de ese objeto en el futuro, el predictor ya no puede hacer trampa usando el pasado inmediato del objeto; *está obligado a inferir la dinámica basándose en cómo los demás objetos visibles y las variables auxiliares (acciones del robot) han interactuado con él*.

## Consecuencias Teóricas
El paper demuestra que este enmascaramiento actúa como una **intervención latente sobre la observabilidad**, induciendo un sesgo inductivo causal ("Causal Inductive Bias"). Formaliza la noción de *Influence Neighborhoods*, probando matemáticamente que la única forma de que el modelo minimice la pérdida es aprendiendo las dependencias condicionales verdaderas de la interacción entre entidades.

## Resultados Empíricos
- **Razonamiento Visual**: En el dataset CLEVRER, mejora las respuestas a preguntas contrafactuales en un ~20% absoluto.
- **Planificación Eficiente**: Al operar sobre unos pocos "slots" de objetos en lugar de miles de parches, C-JEPA reduce el espacio de tokens al 1%, logrando una velocidad de planificación **8 veces mayor** que los modelos basados en parches (como DINO-WM) en tareas de control robótico como Push-T, sin pérdida de rendimiento.

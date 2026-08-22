---
title: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
authors: [Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero]
year: 2026
tags: [paper, jepa, world-models, end-to-end, sigreg]
---

# LeWorldModel: Stable End-to-End JEPA from Pixels

## Resumen Ejecutivo
Este paper introduce **LeWorldModel (LeWM)**, una arquitectura JEPA (Joint-Embedding Predictive Architecture) que se entrena de manera estable de principio a fin (*end-to-end*) directamente desde píxeles en bruto. La innovación principal es que elimina la necesidad de heurísticas complejas para evitar el colapso de representaciones (como *stop-gradients* o *exponential moving averages* usados típicamente en V-JEPA o I-JEPA), utilizando en su lugar una regularización simple basada en la distribución Gaussiana llamada **[[SIGReg]]**.

## Arquitectura (Ver [[LeWorldModel]])
LeWM consta de dos componentes principales:
1. **Encoder ($enc_\theta$)**: Basado en un Vision Transformer (ViT). Mapea observaciones de fotogramas (píxeles) a un espacio latente compacto.
2. **Predictor ($pred_\phi$)**: Un Transformer que modela la dinámica del entorno. Predice el estado latente futuro $\hat{z}_{t+1}$ a partir del estado latente actual $z_t$ y una acción $a_t$.

## Funciones de Pérdida (Ver [[LeWM_Loss]])
El entrenamiento se basa en la suma de dos términos:
1. **Prediction Loss ($L_{pred}$)**: Error cuadrático medio (MSE) entre el embedding predicho y el embedding real codificado del siguiente fotograma.
2. **Regularization Loss ([[SIGReg]])**: Evita el colapso forzando a que las representaciones latentes sigan una distribución Gaussiana isotrópica.

$$ \mathcal{L}_{LeWM} = \mathcal{L}_{pred} + \lambda \text{SIGReg}(Z) $$

## Planificación Latente
Una vez entrenado el modelo del mundo, la planificación para tareas de control se realiza optimizando una secuencia de acciones para minimizar el costo latente entre el estado predicho y el estado latente objetivo. Utilizan el **Cross-Entropy Method (CEM)** y **Model Predictive Control (MPC)**.

## Resultados Clave
- LeWM planifica hasta 48 veces más rápido que modelos fundacionales (ej. DINO-WM).
- Logra entrenarse con solo ~15M de parámetros en una sola GPU en unas pocas horas.
- Demuestra "Physical Understanding" (Entendimiento Físico) al asignar mayor nivel de "sorpresa" a violaciones de la física (ej. un objeto teletransportándose) en el entorno evaluado, en comparación con simples perturbaciones visuales (cambios de color).

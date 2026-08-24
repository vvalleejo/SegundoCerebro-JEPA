---
title: "GeniWorld"
tags: [entity, architecture, world-models, robotics, visual-actions, flow-matching, generative]
---

# GeniWorld

## Descripción
**GeniWorld** es una arquitectura de Modelo de Mundo interactivo para robótica (Tsinghua University & Tencent Robotics X, 2026) que desacopla la cinemática del robot de la dinámica del entorno mediante el uso de **Acciones Visuales (*Visual Actions*)**.

A diferencia de los modelos predictivos latentes puros como [[V-JEPA2]] o [[LeWorldModel]] (que predicen exclusivamente en el espacio de características sin reconstrucción), GeniWorld opera como un **simulador generativo visual de alta fidelidad** capaz de producir predicciones fotograma a fotograma en tiempo real para interacción con humanos y evaluación/entrenamiento de políticas VLA.

---

## Principios de Diseño

1. **Acciones Visuales via URDF**: Elimina la ambigüedad espacial de los vectores numéricos $a_t \in \mathbb{R}^d$ renderizando la cinemática del manipulador robótico directamente en la vista de la cámara.
2. **Concatenación Latente Espacial**: Concatena los tensores latentes de video y acción $(48 + 48 = 96 \text{ canales})$ en un 3D VAE causal, preservando la correspondencia píxel a píxel.
3. **Causal DiT & Flow Matching**: Emplea un Diffusion Transformer autoregresivo (Wan2.2-TI2V-5B) con causal attention y optimización mediante Flow Matching.
4. **Inferencia Interactiva a ~8 Hz**: Utiliza KV Caching y muestreo acelerado de 5 pasos ODE, permitiendo teleoperación humana y bucle cerrado MPC en tiempo real.

---

## Posicionamiento en el Ecosistema de World Models

| Dimensión | JEPA World Models ([[V-JEPA2]], [[LeWorldModel]]) | Generativos Numéricos ([[Ctrl-World]], IRASim) | **GeniWorld** |
| :--- | :--- | :--- | :--- |
| **Espacio de Predicción** | Latente abstracto (sin píxeles) | Píxeles RGB | Píxeles RGB vía Latentes VAE |
| **Condicionamiento de Acción** | Embeddings escalares / AdaLN | Vectores numéricos / AdaLN | **Acciones Visuales Cinemáticas (URDF)** |
| **Generalización OOD** | Alta en representación | Pobre (colapso por fondo/distractores) | **Muy Alta (Zero-Shot OOD)** |
| **Propósito Principal** | Planificación MPC latente directa | Simulación y evaluación | **Simulación interactiva, Evaluador y Síntesis de Datos** |

---

## Enlaces Relacionados
- Paper: [[2026_GeniWorld]]
- Arquitectura y Diagramas: [[2026_GeniWorld]]
- Formulación Matemática: [[Visual_Action_Flow_Matching]]
- Guía de Síntesis: [[World_Models_PhD_Guide]]

---
title: "Yann LeCun World Models Brain - Index"
tags: [index, moc]
---

# Map of Content (MOC) - Yann LeCun World Models Brain

Este es el punto de entrada principal para la base de conocimientos enfocada en los Modelos de Mundo (World Models) y las arquitecturas JEPA de Yann LeCun.

## 📄 Papers Ingeridos
- [[2026_LpWM]]: "LpWM: A Case for Sparse Representations in World Models" (2026). Demostración teórica y empírica de representaciones latentes dispersas (sparse) para dinámicas simplificadas usando RDMReg.
- [[2026_LeVJEPA]]: "LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics" (2026). Encoder de video eficiente y libre de colapso usando SIGReg y token dropping.
- [[2026_LeWorldModel]]: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" (2026). Arquitectura JEPA end-to-end con SIGReg.
- [[2026_VJEPA]]: "VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models" (2026). Generalización probabilística de JEPA y formulación Bayesiana (BJEPA).
- [[2025_V-JEPA2]]: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" (2025). Modelo masivo de video pre-entrenado y su variante Action-Conditioned para robótica.
- [[2026_V-JEPA2.1]]: "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning" (2026). Mejora sobre V-JEPA 2 para obtener representaciones locales densas y de grano fino.
- [[2026_EB-JEPA]]: "A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures" (2026). Formulación EBM de JEPA y librería educativa con Multistep Rollouts.
- [[2026_Causal-JEPA]]: "Causal-JEPA: Learning World Models through Object-Level Latent Masking" (2026). Enmascaramiento de objetos para inducir razonamiento relacional y causal.
- [[2026_CHARM]]: "Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings" (2026). JEPA multimodal para series temporales guiado por descripciones de sensores.
- [[2026_MJEPA]]: "MJEPA: A Simple and Scalable Joint-Embedding Predictive Architecture for Audio-Visual Learning" (2026). JEPA integrado para audio y video mediante predicción cruzada.
- [[2026_MuSe]]: "Multisensory Continual Learning: Adapting Pretrained Visuomotor Policies to Force" (2026). Adaptación de World Models robóticos a nuevos sensores vía Continual Learning.
- [[2026_Rectified_LpJEPA]]: "Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations" (2026). Representaciones latentes esparzas usando regularización RGG.
- [[2023_I-JEPA]]: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023). El modelo fundacional original para imágenes estáticas.
- [[2026_Semantic_Tube]]: "Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA" (2026). JEPA para LLMs usando la hipótesis geodésica.
- [[2026_TC-JEPA]]: "Text-Conditional JEPA for Learning Semantically Rich Visual Representations" (2026). Condicionamiento de texto para la predicción visual de grano fino.
- [[2026_AdaJEPA]]: "AdaJEPA: An Adaptive Latent World Model" (2026). Adaptación a tiempo de prueba (TTA) para modelos de mundo latentes en bucle cerrado MPC.
- [[2026_SkyJEPA]]: "SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors" (2026). Modelo de mundo latente con prober de física diferenciable para control en tiempo real de drones.
- [[2026_LeJEPA_Identifiability]]: "When Does LeJEPA Learn a World Model?" (2026). Demostración matemática formal de la identificabilidad lineal y unicidad Gaussiana en JEPAs.
- [[2025_PLDM]]: "Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models" (2025). Estudio comparativo y método PLDM para planificación con modelos de mundo JEPA offline sin recompensas.
- [[2026_HP-JEPA]]: "HP-JEPA: Hierarchical Partitioning for Multi-Resolution Graph Joint-Embedding Predictive Learning" (2026). JEPA multirresolución en grafos mediante particionamiento jerárquico y readout adaptativo.
- [[2026_Music-JEPA]]: "Music-JEPA: Learning a World Model of Sound from Action" (2026). Modelo de mundo musical condicionado por acciones (audio como estado, pianoroll como acción) y transcripción vía planificación.

- [[2026_GeniWorld]]: "GeniWorld: A Generalizable Interactive World Model for Robotic Manipulation via Visual Actions" (2026). Modelo de mundo interactivo con Flow Matching y acciones visuales renderizadas vía URDF.

## 🧮 Conceptos Matemáticos y Teóricos (Math & Concepts)
- [[RDMReg]]: Rectified Distribution Matching Regularization. Regularizador que aproxima distribuciones a un objetivo RGG (Laplace Rectificado) para esparcidad.
- [[SIGReg]]: Sketched-Isotropic-Gaussian Regularizer. Término de regularización para evitar el colapso de representación hacia distribuciones de baja dimensionalidad.
- [[LeWM_Loss]]: Objetivo matemático minimalista para el entrenamiento de LeWM (MSE + SIGReg).
- [[VJEPA_Loss]]: Objetivo variacional (NLL + KL Divergence) para modelar incertidumbre aleatoria y evitar modelar ruido distractivo.
- [[Dense_Predictive_Loss]]: Extensión de la pérdida JEPA sobre los tokens de contexto visibles ponderados por distancia.
- [[Invariance_Loss]]: Objetivo matemático (MSE) para predicción local-a-global en JEPAs sin heurísticas (ej. LeVJEPA).
- [[Linear_Identifiability]]: Garantía matemática de que el espacio latente de LeJEPA recupera fielmente las variables reales del mundo hasta una rotación ortogonal.

- [[Visual_Action_Flow_Matching]]: Formulación de Flow Matching condicionado por acciones visuales espaciales limpias para interacción física en modelos de mundo.

## 🏗️ Entidades y Arquitecturas (Entities)
- [[LpWM]]: Arquitectura JEPA de World Models caracterizada por generar representaciones latentes distribuidas y dispersas.
- [[LeWorldModel]]: La arquitectura World Model introducida en 2026, caracterizada por su entrenamiento estable basado exclusivamente en regularización gaussiana.
- [[LeJEPA]]: Arquitectura base matemáticamente garantizada contra el colapso mediante SIGReg.
- [[LeVJEPA]]: Adaptación al video de LeJEPA con block-causal attention y extrema eficiencia computacional.
- [[VJEPA]]: Extensión probabilística que mapea a distribuciones predictivas en lugar de puntos deterministas.
- [[BJEPA]]: Bayesian JEPA, usa "Product of Experts" para fusionar el conocimiento físico y las restricciones de una tarea de manera modular.
- [[V-JEPA2]]: Arquitectura de video fundacional que demostró empíricamente la viabilidad del control MPC en espacios latentes JEPA.
- [[V-JEPA2.1]]: Evolución que logra retener estructura espacial fina (Dense Features) mediante supervisión profunda y pérdida de contexto ponderada.
- [[EB-JEPA]]: Framework de Energy-Based Models y librería de investigación de World Models.
- [[C-JEPA]]: Causal-JEPA, usa arquitecturas centradas en objetos (slots) y enmascara objetos completos para forzar el aprendizaje de interacciones en el predictor.
- [[CHARM]]: Modelo fundacional para series temporales basado en JEPA y condicionado por texto.
- [[MJEPA]]: Arquitectura multimodal que alinea audio y video forzando a una modalidad a predecir la representación de la otra.
- [[MuSe]]: Framework de adaptación multi-sensorial para World Models robóticos sin olvidar tareas previas.
- [[Rectified_LpJEPA]]: Variante JEPA enfocada en representaciones latentes esparzas (con muchos ceros) para mayor eficiencia computacional.
- [[I-JEPA]]: El modelo base de la arquitectura (2023) aplicado puramente a imágenes estáticas con enmascaramiento de bloques espaciales.
- [[Semantic_Tube]]: Regularización tipo JEPA para LLMs que reduce el error forzando a los embeddings a seguir trayectorias "rectas" (geodésicas).
- [[TC-JEPA]]: Arquitectura multimodal que usa cross-attention con texto para predecir características visuales.
- [[AdaJEPA]]: Framework de adaptación a tiempo de prueba (Test-Time Adaptation) en bucle cerrado MPC para modelos de mundo.
- [[SkyJEPA]]: Modelo de mundo latente para cuadricópteros con prober de física diferenciable e integración en tiempo real con MPPI.
- [[Linear_Identifiability]]: Marco teórico que demuestra cuándo y por qué LeJEPA recupera la verdadera estructura latente del mundo.
- [[PLDM]]: Método de modelo de mundo latente JEPA para planificación offline sin recompensas con alta generalización OOD.
- [[HP-JEPA]]: Framework de JEPA jerárquico para grafos que captura información estructural a múltiples escalas de resolución.
- [[Music-JEPA]]: Modelo de mundo acústico y musical condicionado por acciones instrumentales que permite transcripción por planificación inversa.

- [[GeniWorld]]: Modelo de mundo interactivo autorregresivo para robótica que desacopla cinemática y dinámica ambiental mediante acciones visuales.

## 🏛️ Arquitecturas y Diagramas (Architecture)
- [[2026_LpWM]]: Arquitectura LpWorldModel con RepReLU y predictores simplificados mediante RDMReg.
- [[2023_I-JEPA]]: Arquitectura I-JEPA con enmascaramiento multi-bloque y predicción en espacio de representación.
- [[2025_PLDM]]: Arquitectura PLDM para aprendizaje de dinámicas latentes y planificación offline sin recompensas.
- [[2025_V-JEPA2]]: Arquitectura V-JEPA 2 / V-JEPA 2-AC para modelos de video y control MPC en robótica.
- [[2026_AdaJEPA]]: Arquitectura AdaJEPA con adaptación a tiempo de test (TTA) en bucle cerrado MPC.
- [[2026_Causal-JEPA]]: Arquitectura C-JEPA con Slot Attention y enmascaramiento latente a nivel de objeto.
- [[2026_CHARM]]: Arquitectura CHARM multimodal (TCN + Gating) condicionada por descripciones de sensores.
- [[2026_EB-JEPA]]: Arquitectura modular Energy-Based JEPA para imágenes, video y control.
- [[2026_HP-JEPA]]: Arquitectura HP-JEPA con particionamiento jerárquico multirresolución en grafos.
- [[2026_LeJEPA_Identifiability]]: Arquitectura y teoría de identificabilidad lineal y unicidad gaussiana en LeJEPA.
- [[2026_LeWorldModel]]: Arquitectura LeWorldModel end-to-end desde píxeles con regularización SIGReg.
- [[2026_LeVJEPA_Arch]]: Arquitectura de LeVJEPA con token dropping uniforme y ausencia de predictor.
- [[2026_MJEPA]]: Arquitectura MJEPA con encoder y predictor compartidos para audio y video.
- [[2026_MuSe]]: Arquitectura MuSe para adaptación continua multisensorial visomotora a sensores de fuerza/par.
- [[2026_Music-JEPA]]: Arquitectura Music-JEPA conectando estados acústicos y acciones de pianoroll.
- [[2026_Rectified_LpJEPA]]: Arquitectura Rectified LpJEPA con ReLU y regularización RGG de máxima entropía.
- [[2026_Semantic_Tube]]: Arquitectura Semantic Tube para fine-tuning geométrico y eficiente en LLMs.
- [[2026_SkyJEPA]]: Arquitectura SkyJEPA con Differentiable Physics Prober y MPPI a 100 Hz para drones.
- [[2026_TC-JEPA]]: Arquitectura TC-JEPA con condicionamiento textual de grano fino vía Cross-Attention.
- [[2026_V-JEPA2.1]]: Arquitectura V-JEPA 2.1 con 3D RoPE y pérdida de contexto ponderada para características densas.
- [[2026_VJEPA]]: Arquitecturas VJEPA (variacional) y BJEPA (bayesiana con Product of Experts) como modelos de mundo probabilísticos.

- [[2026_GeniWorld]]: Arquitectura GeniWorld con renderizador URDF, codificación 3D VAE concatenada y DiT causal con Flow Matching.

## 🧠 Síntesis (Synthesis)
- [[World_Models_PhD_Guide]]: Guía comprensiva para el Doctorado integrando las lecciones de la arquitectura JEPA en diversas modalidades (Visión, Robótica, Series Temporales, Lenguaje).

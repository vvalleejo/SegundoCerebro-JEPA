---
title: "World Models PhD Guide"
tags: [synthesis, moc, phd-guide, jepa]
---

# Guía de Síntesis para el Doctorado en World Models

Esta página conecta de manera jerárquica y temática los distintos avances y arquitecturas basados en Joint-Embedding Predictive Architectures (JEPA), con el fin de proporcionar un mapa mental claro para tu doctorado en modelos de mundo.

## 1. El Marco Teórico y Matemático Base
Todo el ecosistema se sustenta en la idea de que los World Models no deben modelar todos los detalles del mundo a nivel de píxel o sensor crudo, sino centrarse en extraer una representación abstracta predecible.

- **[[I-JEPA]] (2023)**: Demostró que predecir bloques contiguos en el espacio latente aprende representaciones más semánticas que la reconstrucción (como MAE), sin requerir "data augmentations" artificiales.
- **[[SIGReg]]**: Para evitar el colapso (que el predictor siempre emita el mismo valor), en lugar de usar técnicas contrastivas, se introdujo regularizar la covarianza y varianza forzando características que asemejen una Gaussiana isotrópica.
- **[[Linear_Identifiability]] / [[2026_LeJEPA_Identifiability]] (2026)**: Demostración matemática formal (verificada en Lean 4) de que la combinación de alineación y regularización Gaussiana ([[SIGReg]]) permite recuperar linealmente los verdaderos grados de libertad del mundo ($h(z) = Qz$), y prueba que la Gaussiana es la *única* distribución latente con esta garantía.
- **[[Rectified_LpJEPA]] (2026)**: Va un paso más allá de SIGReg, regularizando hacia una distribución Rectified Generalized Gaussian (RGG) para forzar **esparsidad** ($L_0$ norm controlable) y características no negativas, imitando la eficiencia del cerebro humano.

## 2. Modelado de Mundo Visual y Físico (Videos)
Las bases de I-JEPA se expandieron para entender la dinámica temporal y las leyes de la física a través de videos.

- **[[V-JEPA2]]**: Permitió el control MPC (Model Predictive Control) directo desde el espacio latente. 
- **[[V-JEPA2.1]]**: Mejoró el modelo anterior incorporando una pérdida ponderada de contexto (Weighted Context Loss) y capas profundas con supervisión densa para no perder los pequeños detalles espaciales necesarios en robótica de precisión.
- **[[LeWorldModel]] (2026)**: Es la cristalización "end-to-end" de estos principios, logrando un entrenamiento estable sin complejas arquitecturas objetivo asimétricas (solo dependiendo de SIGReg).
- **[[AdaJEPA]] (2026)**: Introduce adaptación a tiempo de prueba (Test-Time Adaptation) en el bucle cerrado de MPC. Permite recalibrar el modelo de mundo latente sobre la marcha tras cada acción ejecutada sin requerir demostraciones de expertos ni datos de recompensa, superando el colapso por desvío de distribución (distribution shift).
- **[[PLDM]] / [[2025_PLDM]] (2025)**: Demuestra empíricamente que planificar con un modelo de mundo JEPA (reconstruction-free) sobre datos offline sin recompensa es superior a los métodos RL model-free en generalización a mapas/geometrías no vistas y transferencia multitarea.

## 3. Extensión a la Probabilidad y Energía (EBMs)
Para modelar entornos con dinámicas inherentemente inciertas o estocásticas (donde una misma acción puede tener múltiples futuros válidos), los JEPAs deterministas se quedaron cortos.

- **[[VJEPA]]**: Variational JEPA introduce variables latentes estocásticas para predecir **distribuciones** de futuros plausibles.
- **[[EB-JEPA]]**: Energy-Based JEPA re-contextualiza JEPA dentro del marco de modelos basados en energía, utilizando variables latentes y una inferencia "Score-Matching". 
- **[[BJEPA]]**: Bayesian JEPA permite un condicionamiento modular (Product of Experts) para fusionar conocimiento empírico visual con reglas físicas a priori.

## 4. Modelado Multisensorial y Robótica
Los humanos no solo usan los ojos; un verdadero modelo de mundo robótico requiere tacto, sonido, fuerza y descripciones semánticas.

- **[[C-JEPA]] (Causal-JEPA)**: Cambió el paradigma de "enmascarar parches" a "enmascarar objetos enteros" (basado en representaciones *object-centric* o slots). Esto fuerza al modelo a inferir las reglas de interacción entre objetos.
- **[[MJEPA]]**: Puso el audio y el video en un mismo codificador unificado. Demostró que predecir de forma cruzada (ej. predecir la representación del sonido a partir de las imágenes) crea un espacio latente semánticamente rico.
- **[[CHARM]]**: Llevó JEPA a series temporales industriales, condicionando la red temporal mediante el texto de descripción de los sensores, logrando un *Zero-Shot* en datos industriales con mucho ruido.
- **[[TC-JEPA]]**: Agregó condicionamiento de texto de grano fino directamente al predictor de JEPA usando cross-attention, mejorando drásticamente el razonamiento denso local.
- **[[MuSe]]**: Abordó cómo integrar *nuevos sensores* (ej. fuerza/torque) a una política de World Model existente sin sufrir de "olvido catastrófico" (Continual Learning).
- **[[SkyJEPA]] (2026)**: Aplicó JEPA al control ágil de drones en tiempo real a alta frecuencia ($>100\text{ Hz}$) mediante un *Physics-Inspired Prober* que traduce latentes a estados físicos $(p, v, R, \omega)$ usando integradores cinemáticos diferenciables, logrando transferencia *Sim-to-Real Zero-Shot*.
- **[[Music-JEPA]] / [[2026_Music-JEPA]] (2026)**: Modela el sonido como un sistema dinámico interactivo (audio como estado, eventos de pianoroll y pedal como acción), demostrando que la predicción latente permite resolver problemas inversos de transcripción mediante planificación en el espacio de acciones.

- **[[GeniWorld]] / [[2026_GeniWorld]] (2026)**: Resuelve el problema de generalización fuera de distribución (OOD) y el acoplamiento espurio en modelos de mundo generativos sustituyendo vectores numéricos por **Acciones Visuales** renderizadas con cinemática URDF. Combina un DiT causal (Wan2.2) y Flow Matching para actuar como simulador interactivo en tiempo real (~8 Hz) para evaluación robusta de políticas VLA ($\\pi_0$) y síntesis masiva de trayectorias.

## 5. Estructuras Relacionales, Grafos y Multirresolución
Los entornos físicos y relacionales complejos no siempre se estructuran en cuadrículas euclidianas (como imágenes o audio), sino en redes y topologías no euclidianas (moléculas, interacciones multi-agente, conectividad abstracta).

- **[[HP-JEPA]] / [[2026_HP-JEPA]] (2026)**: Extiende JEPA a grafos superando el sesgo de granularidad fija mediante un particionamiento jerárquico de grueso a fino ($K_\ell = 2^\ell$). Realiza predicción latente hacia el hiperboloide de Lorentz y permite un readout con ponderación adaptativa por tarea, capturando desde motivos locales hasta topologías globales.

## 6. Salto al Lenguaje y Razonamiento Lógico (LLMs)
Finalmente, los principios de World Models han demostrado desafiar la concepción tradicional del NLP.

- **[[Semantic_Tube]]**: Partiendo de la hipótesis de que los estados ocultos de un LLM al generar texto trazan una ruta "recta" (geodésica) en el colector semántico, proponen una regularización tipo JEPA. Esto mejoró el Signal-to-Noise Ratio y rompió los límites esperados de eficiencia de datos dictados por las leyes de escalado de Chinchilla (usando 16 veces menos datos).

## Siguientes Pasos
Este repositorio está ahora completamente interconectado y te servirá de base. Podrás acceder a los resúmenes y a los conceptos clave usando la sintaxis `[[nombre]]` en el propio Obsidian para navegar ágilmente.

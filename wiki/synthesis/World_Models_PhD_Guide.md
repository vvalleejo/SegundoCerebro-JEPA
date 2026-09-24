---
title: "World Models PhD Guide"
type: "synthesis"
tags: [synthesis, moc, phd-guide, jepa]
---

# Guía de Síntesis para el Doctorado en World Models

Esta página conecta de manera jerárquica y temática los distintos avances y arquitecturas basados en Joint-Embedding Predictive Architectures (JEPA), con el fin de proporcionar un mapa mental claro para tu doctorado en modelos de mundo.

## 1. El Marco Teórico y Matemático Base
Todo el ecosistema se sustenta en la idea de que los World Models no deben modelar todos los detalles del mundo a nivel de píxel o sensor crudo, sino centrarse en extraer una representación abstracta predecible.

- **[[I-JEPA]] (2023)**: Demostró que predecir bloques contiguos en el espacio latente aprende representaciones más semánticas que la reconstrucción (como MAE), sin requerir "data augmentations" artificiales.
- **[[LeJEPA]] / [[2025_LeJEPA]] (2025)**: Punto de inflexión teórico que demostró que la **Gaussiana isotrópica** es la distribución óptima que minimiza el sesgo y la varianza downstream ([[Isotropic_Gaussian_Optimality]]), introduciendo **[[SIGReg]]** para eliminar por completo heurísticas como redes teacher con EMA, stop-gradients o predictores asimétricos. Logró además que la pérdida de entrenamiento correlacione hasta un 99% con la precisión de evaluación lineal.
- **[[SIGReg]]**: Evita el colapso, es decir, que el encoder mapee todo a una constante que el predictor reproduce trivialmente. **No** regulariza covarianza y varianza, como hace VICReg. Impone la **distribución completa** $\mathcal N(0,I)$ con el test de Epps–Pulley sobre la función característica empírica de proyecciones 1D aleatorias (Cramér–Wold). El coste es lineal en batch y dimensión.
- **[[Linear_Identifiability]] / [[2026_LeJEPA_Identifiability]] (2026)**: Demostración matemática formal (verificada en Lean 4) de que la combinación de alineación y regularización Gaussiana ([[SIGReg]]) permite recuperar linealmente los verdaderos grados de libertad del mundo ($h(z) = Qz$), y prueba que la Gaussiana es la *única* distribución latente con esta garantía.
- **[[Rectified_LpJEPA]] (2026)**: Va un paso más allá de SIGReg, regularizando hacia una distribución Rectified Generalized Gaussian (RGG) para forzar **esparsidad** ($L_0$ norm controlable) y características no negativas, imitando la eficiencia del cerebro humano.
- **[[LpWM]] / [[2026_LpWM]] (2026)**: Cuestiona la necesidad de representaciones densas en JEPAs. Utiliza RDMReg para inducir características no negativas y dispersas, demostrando que la dispersión simplifica radicalmente las dinámicas latentes, permitiendo usar predictores de muy baja capacidad (ej. modelos LTI lineales) para planificar con éxito en entornos complejos, revelando además dinámicas factorizadas donde el soporte codifica el régimen de contacto.

## 2. Modelado de Mundo Visual y Físico (Videos)
Las bases de I-JEPA se expandieron para entender la dinámica temporal y las leyes de la física a través de videos.

- **[[V-JEPA2]]**: Permitió el control MPC (Model Predictive Control) directo desde el espacio latente. 
- **[[V-JEPA2.1]]**: Mejoró el modelo anterior incorporando una pérdida ponderada de contexto (Weighted Context Loss) y capas profundas con supervisión densa para no perder los pequeños detalles espaciales necesarios en robótica de precisión.
- **[[LeVJEPA]] / [[2026_LeVJEPA]] (2026)**: Traslada el paradigma LeJEPA al preentrenamiento de video. Elimina predictores y redes teacher, adoptando **block-causal attention** (causal a través del tiempo) y descartando aleatoriamente hasta el 95% de los tokens (*uniform token dropping*), reduciendo drásticamente los FLOPs de preentrenamiento y permitiendo extender representaciones cuadro a cuadro sin re-codificación.
- **[[LeWorldModel]] (2026)**: Es la cristalización "end-to-end" de estos principios, logrando un entrenamiento estable sin complejas arquitecturas objetivo asimétricas (solo dependiendo de SIGReg).
- **[[AdaJEPA]] (2026)**: Introduce adaptación a tiempo de prueba (Test-Time Adaptation) en el bucle cerrado de MPC. Permite recalibrar el modelo de mundo latente sobre la marcha tras cada acción ejecutada sin requerir demostraciones de expertos ni datos de recompensa, superando el colapso por desvío de distribución (distribution shift).
- **[[PLDM]] / [[2025_PLDM]] (2025)**: Demuestra empíricamente que planificar con un modelo de mundo JEPA (reconstruction-free) sobre datos offline sin recompensa es superior a los métodos RL model-free en generalización a mapas/geometrías no vistas y transferencia multitarea.
- **[[C-JEPA]] (Causal-JEPA, 2026)**: Pasa de "enmascarar parches" a "enmascarar objetos enteros" sobre slots *object-centric* de un encoder congelado. Esto obliga al predictor, un transformer bidireccional, a inferir las interacciones entre objetos: +21 puntos en contrafactuales de CLEVRER y planificación >8× más rápida que DINO-WM.
- **[[MotionJEPA]] (2026)**: Identifica el *temporal feature collapse*: el encoder ignora lo que cambia rápido porque el fondo estático ya minimiza la pérdida. Lo corrige con [[DISReg]], que exige predecir el embedding de la imagen diferencia $o_{t+1}-o_t$.
- **[[SG-JEPA]] (Semigroup-JEPA, 2026)**: Extiende LeWM condicionando por la gravedad y entrenando con un **rollout autorregresivo de K pasos con descuento**, sin stop-gradient. La ley de composición del semigrupo surge por construcción, no de una pérdida explícita. Aporta una teoría de defecto de clausura para la generalización OOD en física ([[Latent_Dynamics_Consistency]]).

## 3. Extensión a la Probabilidad y Energía (EBMs)
Para modelar entornos con dinámicas inherentemente inciertas o estocásticas (donde una misma acción puede tener múltiples futuros válidos), los JEPAs deterministas se quedaron cortos.

- **[[VJEPA]]**: Variational JEPA predice una **distribución** $p_\phi(Z_T\mid Z_C,\xi_T)$ sobre el embedding objetivo, en lugar de un punto, con KL frente a un prior fijo. $\xi_T$ es información estructural del target, no una latente estocástica. Su único experimento es un toy lineal ("Noisy TV").
- **[[EB-JEPA]]**: Librería educativa de Meta FAIR que presenta JEPA como modelo basado en energía, $\mathcal E(x,y)=\|P(E(x))-E(y)\|_2^2$. Trae ejemplos de imagen, vídeo y control con VICReg o SIGReg, sin EMA, y planificación MPPI/CEM. No introduce un método nuevo.
- **[[BJEPA]]**: Bayesian JEPA permite un condicionamiento modular (Product of Experts) para fusionar conocimiento empírico visual con reglas físicas a priori.

## 4. Modelado Multisensorial y Robótica
Los humanos no solo usan los ojos; un verdadero modelo de mundo robótico requiere tacto, sonido, fuerza y descripciones semánticas.

- **[[MJEPA]]**: Puso el audio y el video en un mismo codificador unificado. Demostró que predecir de forma cruzada (ej. predecir la representación del sonido a partir de las imágenes) crea un espacio latente semánticamente rico.
- **[[CHARM]]**: Llevó JEPA a series temporales industriales, condicionando la red temporal mediante el texto de descripción de cada canal (CHARM = *Channel-Aware Representation Model*). Con linear probe es competitivo en forecasting, clasificación y anomalías.
- **[[TC-JEPA]]**: Agregó condicionamiento de texto de grano fino directamente al predictor de JEPA usando cross-attention, mejorando drásticamente el razonamiento denso local.
- **[[MuSe]]**: Abordó cómo integrar *nuevos sensores* (ej. fuerza/torque) a una política de World Model existente sin sufrir de "olvido catastrófico" (Continual Learning). No es una JEPA: es una política generativa basada en UVA, y la palabra "JEPA" no aparece en el PDF.
- **[[SkyJEPA]] (2026)**: Aplicó JEPA al control ágil de drones en tiempo real a alta frecuencia ($>100\text{ Hz}$) mediante un *Physics-Inspired Prober* que traduce latentes a estados físicos $(p, v, R, \omega)$ usando integradores cinemáticos diferenciables, logrando transferencia *Sim-to-Real Zero-Shot*.
- **[[Music-JEPA]] / [[2026_Music-JEPA]] (2026)**: Modela el sonido como un sistema dinámico interactivo (audio como estado, eventos de pianoroll y pedal como acción), demostrando que la predicción latente permite resolver problemas inversos de transcripción mediante planificación en el espacio de acciones.

- **[[GeniWorld]] / [[2026_GeniWorld]] (2026)**, *contraste generativo, no JEPA*: resuelve el problema de generalización fuera de distribución (OOD) y el acoplamiento espurio en modelos de mundo generativos sustituyendo vectores numéricos por **Acciones Visuales** renderizadas con cinemática URDF. Combina un DiT causal (Wan2.2) y Flow Matching para actuar como simulador interactivo en tiempo real (~8 Hz) para evaluación robusta de políticas VLA ($\pi_0$) y síntesis masiva de trayectorias.

## 5. Estructuras Relacionales, Grafos y Multirresolución
Los entornos físicos y relacionales complejos no siempre se estructuran en cuadrículas euclidianas (como imágenes o audio), sino en redes y topologías no euclidianas (moléculas, interacciones multi-agente, conectividad abstracta).

- **[[HP-JEPA]] / [[2026_HP-JEPA]] (2026)**: Extiende JEPA a grafos superando el sesgo de granularidad fija mediante un particionamiento jerárquico de grueso a fino ($K_\ell = 2^\ell$). Realiza predicción latente hacia el hiperboloide de Lorentz y permite un readout con ponderación adaptativa por tarea, capturando desde motivos locales hasta topologías globales.

## 6. Salto al Lenguaje y Razonamiento Lógico (LLMs)
Finalmente, los principios de World Models han demostrado desafiar la concepción tradicional del NLP.

- **[[Semantic_Tube]]**: Partiendo de la hipótesis de que los estados ocultos de un LLM al generar texto trazan una ruta "recta" (geodésica) en el colector semántico, proponen una regularización tipo JEPA. Esto mejoró el Signal-to-Noise Ratio y rompió los límites esperados de eficiencia de datos dictados por las leyes de escalado de Chinchilla (usando 16 veces menos datos).

## Siguientes Pasos
Este repositorio está ahora completamente interconectado y te servirá de base. Podrás acceder a los resúmenes y a los conceptos clave usando wikilinks en el propio Obsidian para navegar ágilmente.

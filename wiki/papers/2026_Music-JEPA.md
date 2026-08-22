---
title: "Music-JEPA: Learning a World Model of Sound from Action"
authors: [Ziyu Wang, Kun Fang, Yann LeCun (NYU, McGill, MBZUAI, CIRMMT, AMI Labs)]
year: 2026
tags: [paper, music-jepa, jepa, audio, music, world-models, action-conditioned, planning]
---

# Music-JEPA: Modelo de Mundo Musical Condicionado por Acciones

## Resumen Ejecutivo
Este paper (Julio 2026) presenta **Music-JEPA**, un modelo de mundo que formula la comprensión y generación musical como un **sistema dinámico condicionado por acciones**. A diferencia de los métodos previos que aplican JEPA al audio de forma pasiva (enmascaramiento aleatorio de espectrogramas), Music-JEPA trata el **audio como estado** ($s_t$) y la **ejecución instrumental (pianoroll y pedal de resonancia)** como la **acción** ($a_{t+1}$), aprendiendo a predecir la evolución acústica en el espacio latente.

## El Problema: Aprendizaje Pasivo vs. Interactivo en Audio/Música
1. **Modelos Pasivos**: Los enfoques de autosupervisión en audio (Audio-MAE, A-JEPA, MERT) operan sobre observaciones acústicas sin modelar la causalidad física de cómo las acciones mecánicas (pulsar teclas, pisar pedales) producen el sonido.
2. **Falta de Conceptos Musicales**: Los modelos generativos clásicos reproducen patrones estadísticos superficiales, pero carecen de una comprensión dinámica de tono, tempo, armonía y estructura compositiva.

## Arquitectura y Formulación de Music-JEPA
El framework modela la dinámica latente conjunta entre estados de audio y acciones MIDI en clips de 2 segundos:

$$s_{t+1} = f(s_t, a_{t+1}), \quad a_{t+1} = g(a_t)$$

1. **Representación de Datos**:
   - **Estado $x_t$**: Espectrograma log-mel ($229$ mel bins, $200$ frames a $10\text{ ms}$).
   - **Acción $y_t$**: Pianoroll $y_t^{\text{note}} \in \mathbb{R}^{200 \times 88}$ (velocidad por tono) + señal continua del pedal de resonancia (*sustain pedal*) $y_t^{\text{pedal}} \in \mathbb{R}^{200}$.
2. **Encoders y Predictores ViT**:
   - **State Encoder $\mathcal{E}_s$** y **Action Encoder $\mathcal{E}_a$**: Vision Transformers que procesan parches de tiempo-frecuencia y tiempo-tono.
   - **State Predictor $f(s, a)$**: ViT con *cross-attention* a los tokens de acción $a_{t+1}$ en cada capa para predecir el estado latente $s_{t+1}$.
   - **Action Predictor $g(a)$**: Transformer que modela la estructura temporal previa de las acciones $a_{t+1}$.
3. **Objetivo de Entrenamiento**:
   $$\mathcal{L}(\theta) = \|f(s_t, a_{t+1}) - s_{t+1}\|^2 + \lambda \|g(a_t) - a_{t+1}\|^2$$
   Prevención del colapso mediante codificadores objetivo actualizados por Media Móvil Exponencial (**EMA**) y normalización por capas (LayerNorm).

## Planificación Inversa: Transcripción como Búsqueda de Acciones
Music-JEPA permite la **transcripción de piano mediante planificación latente** (inferir la secuencia de acciones $a_{1:T}^*$ que mejor explica un audio objetivo $s_{1:T}$):

$$a_{1:T}^* = \arg\min_{a_{1:T}} \sum_{t=1}^{T-1} \left( \|f(s_t, a_{t+1}) - s_{t+1}\|^2 + \|g(a_t) - a_{t+1}\|^2 \right)$$

Para evitar la inestabilidad de la optimización por gradiente en espacios de alta dimensión, entrenan un predictor inverso amortizado $\hat{a}_{t+1} = h(s_t, s_{t+1}, a_t)$ acoplado a un decodificador de acciones.

## Resultados
- **Dinámica Causal y Contrafactual**: Alta sensibilidad a perturbaciones tonales y temporales; al cambiar la acción a una contrafactual (ej. arpegio o escala), el espectrograma imaginado varía coherentemente.
- **Tareas Downstream (MIR)**: Supera a los modelos JEPA de solo audio (AO-JEPA) en *beat tracking*, identificación de compositores y reconocimiento de tonalidad, compitiendo con MERT (95M params) usando solo **19M de parámetros** (7% del tamaño).
- **Control de Pedal Continuo**: Logra el estado del arte en estimación precisa y continua de la curva de pedal de resonancia, superando a métodos supervisados tradicionales.

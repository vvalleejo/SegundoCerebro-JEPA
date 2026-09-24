---
title: "MotionJEPA: Preventing Temporal Feature Collapse by Capturing Visual Changes in Latent Space"
authors: [Markus Karmann, Shile Li, Christian Internò, Bruno Andreis, David Klindt, Randall Balestriero, Jindong Gu, Philip Torr, Qi Zhang, Peng-Tao Jiang, Hao Zhang, Bo Li, Onay Urfalioglu (University of Oxford, vivo Tech Research, Bielefeld University, Brown University, AMI Labs)]
year: 2026
venue: "arXiv preprint"
arxiv: "2609.23881"
source_pdf: "raw/MotionJEPA Preventing Temporal Feature Collapse by Capturing Visual.pdf"
repo: "https://github.com/mkarmann/motion-jepa"
type: "paper"
family: "jepa"
modality: [video, control]
anti_collapse: [disreg, sigreg]
predictor: true
planner: [cem]
tags: [paper, motion-jepa, disreg, jepa, world-models, temporal-collapse, feature-suppression, slow-features, inverse-dynamics, sigreg]
---

# MotionJEPA: Preventing Temporal Feature Collapse by Capturing Visual Changes in Latent Space

## Resumen Ejecutivo

El paper (arXiv, 2026) presenta **MotionJEPA** y su función de regularización central **DISReg** (*Difference Image and Single image embedding Regularization*), desarrollados por investigadores de la Universidad de Oxford, vivo Tech Research, la Universidad de Brown y AMI Labs (con coautoría de Randall Balestriero). El trabajo aborda un fallo teórico y empírico fundamental de las arquitecturas JEPA (*Joint-Embedding Predictive Architectures*) en modelado temporal de mundo: el **sesgo hacia características lentas (*slow-feature bias*) y el colapso temporal de características (*temporal feature collapse*)**.

En los modelos JEPA estándar (como [[LeWorldModel]], [[LeJEPA]] o [[2025_V-JEPA2]]), la función de pérdida predictiva puede satisfacerse trivialmente codificando distractores estáticos o de variación lenta (por ejemplo, fondos fijos, texturas o contadores), suprimiendo por completo la dinámica rápida de primer plano (como pelotas en movimiento, obstáculos o actuadores). Las soluciones previas basadas en dinámica inversa (como SMWM o Delta-JEPA) mitigan este colapso pero **dependen obligatoriamente de etiquetas de acción**, siendo ciegas a dinámicas pasivas del entorno.

MotionJEPA resuelve esto introduciendo **DISReg**, un regularizador visualmente anclado que predice los embeddings de **imágenes de diferencias temporales** ($o_{t+1} - o_t$) a partir de los embeddings de estado latente consecutivos ($z_t, z_{t+1}$) **sin reconstrucción de píxeles ni supervisión de acciones**. Evaluado exhaustivamente en 3 juegos sintéticos con factores de variación desacoplados (Pong, Dino, Golf) y en 4 benchmarks de control de [[LeWorldModel]] con texturas de madera de distracción estática (PushT, TwoRoom, OG-Cube Single, Reacher), MotionJEPA previene de forma consistente el colapso de características, genera trayectorias latentes más rectas y eleva el éxito en planificación MPC (CEM) hasta un **81.8%** (frente al 20.4% de LeWM original).

---

## El Problema: Supresión de Características y Sesgo Hacia Dinámicas Lentas

### 1. El Dilema de la Selección de Características en JEPA
A diferencia de los modelos basados en reconstrucción de píxeles (autoencoders enmascarados, diffusion models), las arquitecturas JEPA aprenden representaciones latentes abstractas prediciendo el embedding del estado futuro:
$$\min_{\theta, \phi} \mathcal{L}_{\text{pred}} = \|\hat{z}_{t+1} - z_{t+1}\|_2^2$$
Para evitar el colapso informacional clásico (donde $z_t$ colapsa a una constante), métodos modernos como [[LeWorldModel]] aplican regularización isotrópica gaussiana ([[SIGReg]]) sobre el batch. Sin embargo, los autores demuestran que **evitar el colapso global de varianza no impide la supresión selectiva de características**.

### 2. Por qué los JEPAs Estándar Prefieren Características Lentas
- Si una escena contiene un distractor estático de alta dimensión (ej. una textura de fondo compleja) y un objeto pequeño que se mueve rápidamente (ej. una pelota de golf o un proyectil):
  - Predecir el fondo estático es trivial: $z_{\text{fondo}, t+1} \approx z_{\text{fondo}, t}$, por lo que el error predictivo cuadrático es prácticamente cero.
  - Predecir la dinámica rápida requiere modelar aceleraciones, colisiones y derivadas temporales de orden superior.
- Como resultado, el encoder minimiza la pérdida predictiva descartando la dinámica rápida y dedicando casi toda su capacidad latente al fondo estático. El modelo sufre un **colapso temporal de características**.

### 3. Insuficiencia de las Alternativas Existentes
- **Modelos de Dinámica Inversa (SMWM, Delta-JEPA)**: Entrenan un predictor $\hat{a}_t = \text{Inv}(z_t, z_{t+1})$ para predecir la acción ejecutada. *Limitación*: Requieren etiquetas de acción y solo capturan grados de libertad controlables por el agente, colapsando ante dinámicas ambientales pasivas (ej. proyectiles autónomos o caídas libres).
- **SIGReg Temporal (LeNEPA / LeWM-Time)**: Aplica la regularización gaussiana a lo largo del eje temporal en lugar del eje del batch. *Limitación*: Destruye la coherencia de trayectorias complejas y fracasa en presencia de múltiples escalas de movimiento.
- **StopGradient en el Target (LeWM-Detached)**: Aplica $\text{sg}[z_{t+1}]$ en el objetivo; sigue colapsando ante distractores visuales estáticos.

---

## Metodología: DISReg y MotionJEPA

### 1. Imagen de Diferencia Temporal
Dada una secuencia de observaciones visuales $o_t, o_{t+1} \in \mathbb{R}^{3 \times H_{\text{obs}} \times W_{\text{obs}}}$, se calcula la diferencia temporal directa en píxeles:
$$o^{\text{diff}}_t = o_{t+1} - o_t$$
En el dominio de la diferencia, **cualquier elemento estático desaparece idénticamente** ($o_{t+1} - o_t = 0$), aislando de forma puramente visual el movimiento y las transiciones del entorno sin necesidad de máscaras semánticas.

### 2. Arquitectura de Módulos de DISReg
MotionJEPA introduce dos módulos auxiliares durante el entrenamiento:
1. **Difference Encoder ($\text{DiffEnc}_\alpha$)**: Mapea la imagen de diferencia a un vector latente de cambio visual $d_t \in \mathbb{R}^{D_d}$:
   $$d_t = \text{DiffEnc}_\alpha(o_{t+1} - o_t)$$
2. **Difference Predictor ($\text{DiffPred}_\beta$)**: Módulo estilo dinámica inversa que predice el embedding de cambio $\hat{d}_t$ a partir de la concatenación de los embeddings de estado $z_t$ y $z_{t+1}$:
   $$\hat{d}_t = \text{DiffPred}_\beta(z_t, z_{t+1})$$

### 3. Función de Pérdida de DISReg
Para evitar que $d_t$ colapse a cero y para mantener balanceadas las componentes estáticas y dinámicas, DISReg formula tres pérdidas acopladas:
$$\mathcal{L}_{\text{DISReg}} = \lambda_z L_z + \lambda_d L_d + \lambda_{\text{pred}} L_{\text{pred}}$$
donde:
- $L_z = \text{SIGReg}(z)$: Regularizador estático que preserva características globales y estructura gaussiana isotrópica.
- $L_d = \text{SIGReg}(d)$: Regularizador dinámico que evita el colapso del espacio de diferencias.
- $L_{\text{pred}} = \text{MSE}(d_t, \hat{d}_t)$: Pérdida de predicción de diferencias latentes.

El objetivo total de MotionJEPA combina la predicción forward con DISReg:
$$\mathcal{L}_{\text{MotionJEPA}} = \mathcal{L}_{\text{DISReg}} + \text{MSE}(\hat{z}_{t+1}, z_{t+1})$$

Hiperparámetros fijos para todos los experimentos: $\lambda_z = 0.25$, $\lambda_d = 2.0$, $\lambda_{\text{pred}} = 0.5$.

> [!IMPORTANT]
> **Sin Sobrecarga en Inferencia**: Los módulos $\text{DiffEnc}_\alpha$ y $\text{DiffPred}_\beta$ se utilizan **únicamente durante el entrenamiento**. En tiempo de despliegue y planificación, ambos se descartan por completo; el modelo de mundo opera exclusivamente con el encoder estándar $\text{Enc}_\theta$ y el predictor forward $\text{Pred}_\phi$.

---

## Resultados Empíricos

### 1. Detección de Colapso en Juegos Sintéticos (Pong, Dino, Golf)
Mediante sondas MLP residuales de alta capacidad entrenadas sobre representaciones congeladas, se evalúa si cada factor de la escena está presente o fue suprimido (NMSE $\le 0.1$ indica retención; $> 0.1$ colapso):
- **Pong**: LeWM colapsa en la pelota y en las palas (NMSE = 0.999 y 1.002), codificando únicamente el marcador estático (0.003). MotionJEPA retiene todos los factores con NMSE $\le 0.005$.
- **Dino**: LeWM retiene el cielo y los corazones pero suprime por completo el cactus en movimiento (0.769). MotionJEPA retiene el cactus (0.011) y el dinosaurio (0.006).
- **Golf**: Con múltiples barras móviles y textura de fondo, todos los baselines (incluyendo LeWM-Flat* y SMWM*) colapsan en las barras móviles con pérdidas $> 0.30$. MotionJEPA es el **único modelo que previene el colapso en todas las categorías**, logrando un NMSE medio de 0.061 frente al 0.763 de LeWM.

### 2. Control y Planificación en Benchmarks de LeWorldModel con Fondos Distractores
En entornos de control robótico (Cube, PushT, Reacher, TwoRoom) a los que se añade un fondo estático de madera muestreado aleatoriamente por episodio:
- **Planificación a 25 pasos (CEM)**:
  - LeWM original cae a una tasa de éxito media de solo **20.4%**.
  - LeWM-Flat alcanza 41.7%.
  - IDM (supervisión de acción inversa) logra 77.2%.
  - **MotionJEPA (standalone)** alcanza **81.8%**.
  - **MotionJEPA + IDM** alcanza el mejor resultado global con **86.4%**.
- **Planificación a 50 pasos (Re-planning CEM)**:
  - LeWM colapsa al **11.8%**.
  - IDM se sitúa en 59.3%.
  - **MotionJEPA** mantiene un **70.3%** de éxito medio (destacando un **98.0% en Reacher** y **63.2% en Cube**).

### 3. Rectitud de Trayectorias Latentes (*Latent Path Straightness*)
Analizando la métrica de curvatura latente $S(z) = \frac{\|z_T - z_0\|_2}{\sum_{t=0}^{T-1} \|z_{t+1} - z_t\|_2}$, MotionJEPA genera trayectorias latentes significativamente más rectas y menos entrelazadas para cada factor físico, facilitando la interpolación y la optimización de trayectorias en CEM.

---

## Relevancia para el Doctorado

MotionJEPA aporta fundamentos clave para la investigación en World Models y JEPA:
1. **Diagnóstico Formal del Sesgo Lento**: Evidencia cómo el objetivo de predicción temporal en espacio latente actúa como un filtro pasa-bajos (*Slow Feature Analysis*), explicando por qué los JEPAs fallan ante fondos y texturas realistas.
2. **Supervisión de Cambio sin Reconstrucción de Píxeles**: Ofrece un mecanismo elegante para obligar al encoder a preservar bits de dinámica visual sin caer en el coste computacional ni en el modelado de ruido de alta frecuencia de los métodos generativos por píxeles.
3. **Complementariedad con Dinámica Inversa y Semigrupos**: DISReg puede combinarse de forma transparente con objetivos de semigrupo ([[Semigroup_Rollout_Consistency]]) y con supervisión de acciones (IDM), proporcionando un marco integral contra el colapso dimensional y temporal.

---

## Referencias Cruzadas
- **Arquitectura**: [[wiki/architecture/2026_MotionJEPA|2026_MotionJEPA]]
- **Arquitectura**: [[2026_MotionJEPA]]
- **Conceptos Matemáticos**: [[DISReg]], [[SIGReg]], [[LeWM_Loss]]
- **Entidades**: [[MotionJEPA]], [[LeWorldModel]], [[2026_Semigroup-JEPA]]
- **Repositorio**: [https://github.com/mkarmann/motion-jepa](https://github.com/mkarmann/motion-jepa)

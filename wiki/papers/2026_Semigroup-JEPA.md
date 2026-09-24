---
title: "Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization"
authors: [Andy Zeyi Liu, Haoran Sun, Lucas Baker, Randall Balestriero, John Sous (Yale University, Jump Trading, Brown University)]
year: 2026
venue: "arXiv preprint"
arxiv: "2609.10464"
source_pdf: "raw/Semigroup-JEPA Latent Dynamics Consistency for.pdf"
repo: "https://github.com/sg-jepa/sg-jepa"
type: "paper"
family: "jepa"
modality: [video, control]
anti_collapse: [sigreg]
predictor: true
planner: [diffusion-policy]
tags: [paper, sg-jepa, jepa, world-models, physics-generalization, semigroup, latent-rollout, sigreg, zero-shot, mujoco]
---

# Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization

## Resumen Ejecutivo

El paper (arXiv:2609.10464, Septiembre 2026) presenta **Semigroup-JEPA (SG-JEPA)**, una extensión condicionada por parámetros físicos del marco [[LeWorldModel]] (LeWM) que aborda una pregunta fundamental para los modelos de mundo basados en JEPA: *¿Aprende realmente el modelo las leyes físicas de la dinámica subyacente cuando sus predicciones locales se componen a lo largo de horizontes temporales extensos, y puede generalizar fuera de distribución (OOD) a leyes físicas no vistas?*

SG-JEPA suministra el parámetro que gobierna la física (específicamente, el campo gravitatorio escalar $g$) al modelo temporal a través del canal de acciones y entrena conjuntamente el encoder visual $e_\phi$ y el predictor $p_\theta$ mediante una **pérdida de rollout autorregresivo latente descontada**, regularizada mediante [[SIGReg]]. Los autores demuestran que las actualizaciones temporales repetidas sobre secuencias sin control externo forman un **semigrupo discreto de evolución** en el espacio latente. Evaluado en 8 entornos MuJoCo (cuerpos rígidos 2D, tiro parabólico 3D y manipulación robótica) entrenados en una banda estrecha de gravedad y testeados en un rango extendido OOD, SG-JEPA reduce el error de predicción en bucle abierto hasta en $2\times$ frente a [[DINO-WM]] y LeWM, y eleva la tasa de éxito de control en bucle cerrado mediante políticas de difusión hasta en $2.5\times$.

Mediante un modelo teórico de características lineales (*linear feature model*), los autores demuestran analíticamente y validan experimentalmente (experimentos de *crossover*) un hallazgo crucial: **la mayor parte de la ganancia procede de la representación que el encoder aprende durante el entrenamiento recurrente (al preservar las características dinámicamente cerradas que el predictor puede propagar), y no meramente de que el predictor aprenda mejores dinámicas**.

---

## El Problema: Predicción Local vs. Composición a Largo Plazo y Generalización Física OOD

Los modelos de mundo basados en JEPA ([[2023_I-JEPA]], [[2025_V-JEPA2]], [[2026_LeWorldModel]], [[DINO-WM]]) han demostrado que predecir en el espacio de incrustación latente en lugar de reconstruir píxeles crudos previene el desperdicio de capacidad en ruido perceptual irrelevante. Sin embargo, adolecían de limitaciones clave:
1. **Entrenamiento Teacher-Forced de Un Solo Paso**: Modelos como LeWM entrenan el predictor siempre alimentado con los embeddings latentes reales codificados del paso previo ($z_t$), no con sus propias predicciones ($\hat{z}_t$). Esto los hace ciegos a la acumulación y amplificación recursiva de errores en horizontes largos.
2. **Leyes de Transición Fijas**: Los modelos previos aprenden dinámicas bajo un único conjunto estático de parámetros del entorno. No existía evidencia de que pudieran extrapolar a diferentes regímenes dinámicos (e.g., adaptar dinámicas terrestres a gravedad lunar o marciana sin reentrenamiento).
3. **Representaciones Congeladas vs. Conjuntas**: DINO-WM utiliza representaciones congeladas de DINOv2, las cuales no están optimizadas para la clausura predictiva de las ecuaciones de movimiento.

---

## Metodología y Arquitectura de Semigroup-JEPA

### 1. Codificación y Condicionamiento de Parámetros Físicos
- **Encoder Visual ($e_\phi$)**: Un Vision Transformer (ViT-Tiny con 12 capas, 3 cabezas de atención, dimensión oculta 192, parches de $8\times 8$ para $128\times 128$ o $16\times 16$ para $256\times 256$). El token `[CLS]` final $h_t \in \mathbb{R}^{192}$ se proyecta mediante un MLP de una capa oculta a $z_t \in \mathbb{R}^{256}$. Se entrena desde inicialización aleatoria de forma conjunta con el predictor.
- **Canal de Acción Aumentado**: La gravedad $g$ se concatena con el vector de control externo $u_t$:
  $$\tilde{a}_t = [u_t;\, g]$$
  Un encoder de acciones $q_\psi$ (convolución temporal seguida de un MLP con activaciones SiLU) mapea $\tilde{a}_t$ a un embedding de contexto $c_t \in \mathbb{R}^{256}$. Si no hay controles externos ($u_t = \emptyset$), $g$ normalizado (z-score) actúa como la única señal de control.
- **Predictor Latente ($p_\theta$)**: Recibe una ventana de historia de longitud $H$ y predice el siguiente estado latente:
  $$\hat{z}_{t+1} = p_\theta(z_{t-H+1:t},\, c_{t-H+1:t})$$
  Se exploran tres familias arquitectónicas:
  - **GRU Residual**: 3 capas residuales de ancho 512, donde el estado oculto se concatena con la acción proyectada, pasa por un MLP y es procesado por una GRU unicapa.
  - **State-Space Model (SSM / Mamba-S6)**: Misma profundidad y anchura, sustituyendo la celda GRU por bloques selectivos S6.
  - **Transformer Causal**: 6 capas y 16 cabezas con inyección de acciones vía Adaptive Layer Normalization (AdaLN), idéntico a LeWM.

### 2. Estructura de Semigrupo y Función de Pérdida de Rollout Descontada
En ausencia de acciones externas para un $g$ fijo, la iteración del operador de actualización define una acción de semigrupo discreto:
$$S_{\theta,g}(k) = F_{\theta,g}^{\circ k}, \quad S_{\theta,g}(0) = I, \quad S_{\theta,g}(k + \ell) = S_{\theta,g}(\ell) \circ S_{\theta,g}(k)$$
A diferencia de LeWM (pérdida MSE a 1 paso con teacher forcing), SG-JEPA realiza un **rollout autorregresivo latente de $K$ pasos**, retroalimentando recursivamente sus propias predicciones latentes:
$$z_s^{\text{roll}} = \begin{cases} z_s & \text{si } s \le t \\ \hat{z}_s & \text{si } s > t \end{cases}$$
$$\hat{z}_{t+k} = p_\theta(z^{\text{roll}}_{t+k-H:t+k-1},\, c_{t+k-H:t+k-1}), \quad k = 1, \dots, K$$

La pérdida de rollout normalizada con descuento exponencial ($\gamma = 0.95$) es:
$$\mathcal{L}_{\text{roll}} = \sum_{k=1}^K w_k \|\hat{z}_{t+k} - z_{t+k}\|_2^2, \quad w_k = \frac{\gamma^{k-1}}{\sum_{j=1}^K \gamma^{j-1}}$$

El objetivo total unificado incorpora regularización isotrópica gaussiana ([[SIGReg]]):
$$\mathcal{L}_{\text{SG-JEPA}} = \mathcal{L}_{\text{roll}} + \lambda_{\text{sig}} \mathcal{L}_{\text{SIGReg}}$$

> [!IMPORTANT]
> **Sin Stop-Gradient en el Encoder Target**: Los estados latentes objetivo $z_{t+k}$ provienen del *mismo encoder entrenable* $e_\phi$ sin operador `stop-gradient` ni teacher EMA heurístico. El colapso dimensional se evita rigurosamente gracias a SIGReg, mientras que retropropagar $\mathcal{L}_{\text{roll}}$ en el encoder entrena a la representación para retener únicamente las componentes de información dinámicamente consistentes bajo composición temporal.

### 3. Optimización Híbrida Muon / AdamW
El modelo se optimiza durante 20 épocas con un esquema híbrido:
- **Muon**: Actualizaciones ortogonalizadas para todos los tensores matriciales 2D (learning rate $1 \times 10^{-4}$).
- **AdamW**: Para vectores 1D, bias y embeddings (learning rate $5 \times 10^{-5}$).
Este optimizador logra una convergencia sustancialmente más rápida y un uso más eficiente de los datos que AdamW puro.

---

## Modelo Teórico de Características Lineales (Linear Feature Model)

Para explicar rigurosamente de dónde surge la ventaja a horizontes largos y bajo gravedad OOD, el paper formula una teoría analítica exhaustiva (detallada en [[Latent_Dynamics_Consistency]]):

1. **Estado de Características y Operador de Transición**:
   Existe un estado no observable $\phi_t \in \mathbb{R}^{d_\phi}$ gobernado por $\phi_{t+1} = T(g)\phi_t + \xi_{t+1}$. El encoder proyecta mediante $W \in \mathbb{R}^{d_z \times d_\phi}$ (con $WW^\top = I_{d_z}$) a $z_t = W \phi_t$.
2. **Defecto de Clausura Predictiva**:
   Se define el defecto medio condicional:
   $$\delta_t(g) = C_W(g)\phi_t + [A_W(g) - \hat{A}(g)]z_t$$
   donde $C_W(g) = W T(g) - A_W(g) W$ es el término de clausura. Si $C_W(g) = 0$, la representación latente es **predictivamente cerrada** (contiene toda la información necesaria para predecir la media del siguiente latente sin depender de las dimensiones descartadas por $W$).
3. **Cota de Extrapolación por Cobertura de la Ley (Law Coverage)**:
   Si la ley física descompone en una base $T(g) = \sum_{k=1}^m \psi_k(g) T_k$ con matriz de momentos $M_\psi = \mathbb{E}_{g \sim P_{\text{tr}}}[\psi(g)\psi(g)^\top]$, el error local en una gravedad no vista $g^\star$ está acotado por el factor de cobertura $L_{\text{law}}(g^\star) = \psi(g^\star)^\top M_\psi^{-1} \psi(g^\star)$:
   $$\|\delta_t(g^\star)\|_2 \le \sqrt{L_{\text{law}}(g^\star)} \left( B_z \epsilon_{\text{op}} + B_\phi \epsilon_{\text{cl}} \right)$$
   Para vuelo libre balístico, $\psi(g) = [1, g]^\top$, por lo que $L_{\text{law}}(g^\star) = 1 + \frac{(g^\star - \mu_{\text{tr}})^2}{\sigma_{\text{tr}}^2}$. El error crece cuadráticamente con la distancia a la media de entrenamiento.
4. **Acumulación de Error por Semigrupo**:
   El error de rollout libre a horizonte $h$ sigue la fórmula telescópica:
   $$e_h = z_h - \hat{z}_h = \sum_{j=0}^{h-1} \hat{A}(g)^{h-1-j} [\delta_j(g) + W \xi_{j+1}]$$
5. **Teorema de Inversión de Ranking (Ranking Reversal Theorem)**:
   Los autores demuestran formalmente que un objetivo de 1 solo paso (teacher-forcing) puede preferir una representación $W_1$ sobre $W_2$, mientras que bajo composición multi-paso $W_2$ exhibe un error radicalmente menor. Entrenar con $\mathcal{L}_{\text{roll}}$ alinea la métrica del gradiente con el semigrupo de evaluación a largo plazo.

---

## Resultados Empíricos

### 1. Predicción Física en Cuerpos Rígidos 2D (MuJoCo)
- **Datasets**: Formas planas (cuadrado, triángulo, pentágono, casa) dentro de una caja con colisiones elásticas, entrenadas en $g \sim \mathcal{N}(4, 0.5^2)$ y evaluadas en 25 gravedades desde $g = -2$ hasta $g = 10$.
- **Rollout de 44 Pasos**:
  - En el cuadrado, SG-JEPA (GRU) reduce el error de posición, velocidad y rotación acumulada entre un **$31\%$ y un $48\%$** respecto a DINO-WM y LeWM original.
  - En gravedad lejana OOD ($g \le 2$ o $g \ge 6$), SG-JEPA reduce el error local a un paso en un **$32\%$** antes de cualquier realimentación recursiva.
  - La composición libre amplifica esta ventaja: la brecha entre modelos se multiplica hasta alcanzar un pico en el horizonte 20.
- **Aislamiento Mecanicista (Crossover Experiment)**: Al congelar los encoders entrenados y ajustar nuevos predictores desde cero (tanto GRU como Transformer), el encoder de SG-JEPA supera sistemáticamente al de Transformer/LeWM por ~12% en error medio de rollout, confirmando que la ventaja reside en la **calidad de la representación del encoder** inducida por el entrenamiento de semigrupo.

### 2. Predicción 3D y Control Robótico en Bucle Cerrado
- **Approach Ball (Proyectil 3D)**: Reducción del error de posición 3D en un **$34\%$** frente a DINO-WM y un **$50\%$** frente a LeWM original a lo largo de 44 pasos.
- **Control Robótico con Diffusion Policy**:
  - **Arm Catcher Ball**: Un brazo robótico atrapa una pelota en vuelo. SG-JEPA eleva la tasa de éxito del **$9.5\%$ (DINO-WM) al $23.3\%$ (SG-JEPA)**.
  - **Franka Paddle-to-Basket**: Un brazo Panda intercepta la pelota y la encesta. Aumenta la conversión efectiva post-impacto del **$27.4\%$ al $30.5\%$**.
  - **Arm Paddle Ball**: Malabarismo continuo con pala inclinable. Eleva el éxito del **$17.7\%$ al $23.8\%$**, manteniendo botes por encima del umbral de $0.28\text{ m}$.

---

## Relevancia para el Doctorado

Este trabajo es de importancia crítica para una tesis en World Models y JEPA:
1. **Unión de Teoría Algebraico-Dinámica y JEPA**: Formaliza la evolución latente mediante operadores de semigrupo ($S_{\theta,g}(k)$) y sienta las bases analíticas de la **clausura predictiva** ($C_W(g) = 0$).
2. **Superación del Teacher-Forcing sin Teacher EMA**: Demuestra que se puede entrenar un encoder de forma end-to-end con pérdidas de predicción recursiva ($K$ pasos) sin stop-gradient ni colapso de representación, siempre que se combine con [[SIGReg]].
3. **Generalización OOD Fundada**: Muestra el protocolo formal para condicionar modelos de mundo por parámetros físicos latentes o explícitos ($g$), caracterizando las cotas de generalización mediante matrices de momentos de la ley física ($M_\psi$).

---

## Referencias Cruzadas
- **Arquitectura**: [[wiki/architecture/2026_Semigroup-JEPA|2026_Semigroup-JEPA]]
- **Arquitectura**: [[2026_Semigroup-JEPA]]
- **Conceptos Matemáticos**: [[Semigroup_Rollout_Consistency]], [[Latent_Dynamics_Consistency]], [[SIGReg]], [[LeWM_Loss]]
- **Entidades**: [[SG-JEPA]], [[LeWorldModel]], [[DINO-WM]]
- **Línea Teórica**: [[Linear_Identifiability]], [[Isotropic_Gaussian_Optimality]]

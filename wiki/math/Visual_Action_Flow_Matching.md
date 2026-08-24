---
title: "Visual-Action Conditional Flow Matching"
tags: [math, flow-matching, generative, world-models, robotics, loss]
---

# Visual-Action Conditional Flow Matching

## 1. Motivación y Formulación General

En los modelos de mundo generativos aplicados a la robótica, condicionar directamente el proceso de difusión mediante vectores escalares de acción $a_t \in \mathbb{R}^d$ genera una desconexión espacial con la cuadrícula de características visuales. 

**Visual-Action Conditional Flow Matching** resuelve este problema mapeando la cinemática articular al espacio euclidiano de la cámara y formulando el problema de aprendizaje de modelos de mundo como una regresión de campos vectoriales en el espacio latente continuo, condicionado por una señal cinemática densa libre de ruido.

---

## 2. Generación Cinemática y Espacio Latente Conjunto

Dado un manipulador robótico con modelo cinemático URDF y una cámara calibrada con matriz de proyección $K_{\text{cam}}$, la secuencia de acciones $a_{t+1:t+H}$ genera la secuencia de movimiento visual:

$$ m_{t+1:t+H} = \mathcal{R}_{\text{URDF}}\left(a_{t+1:t+H}, K_{\text{cam}}\right) \in \mathbb{R}^{H \times 3 \times H_0 \times W_0} $$

Ambas modalidades (observación de video $o$ y acción visual $m$) se mapean mediante un codificador causal 3D VAE $\mathcal{E}_{\text{3D-VAE}}$ a tensores latentes de dimensión $C = 48$:

$$ z_v = \mathcal{E}_{\text{3D-VAE}}(o) \in \mathbb{R}^{C \times L \times H' \times W'}, \quad z_a = \mathcal{E}_{\text{3D-VAE}}(m) \in \mathbb{R}^{C \times L \times H' \times W'} $$

El tensor latente conjunto se construye por concatenación directa de canales:

$$ z = [z_v; z_a] \in \mathbb{R}^{2C \times L \times H' \times W'}, \quad 2C = 96 $$

---

## 3. Dinámica de Flow Matching Condicionado

Sea $z_{t+1} \in \mathbb{R}^{C \times L_b \times H' \times W'}$ el bloque latente de la observación futura objetivo. Se define una interpolación lineal entre una variable aleatoria Gaussiana $\epsilon \sim \mathcal{N}(0, I)$ y el estado objetivo $z_{t+1}$ a lo largo del tiempo continuo de flujo $s \in [0, 1]$:

$$ z_{t+1}^{(s)} = (1 - s)\epsilon + s z_{t+1} $$

Derivando con respecto a $s$, el campo de velocidades objetivo (vector field) del transporte óptimo viene dado por:

$$ \dot{z}_{t+1}^{(s)} \equiv \frac{d}{ds} z_{t+1}^{(s)} = z_{t+1} - \epsilon $$

### Función de Pérdida de GeniWorld
El modelo neuronal parameterized por $\theta$, $v_\theta(\cdot)$, predice este campo de velocidades condicionado por la historia pasada $z_{\le t}$, la instrucción textual $c$, y la **acción visual limpia no difundida** $z_{a, t+1}$:

$$ \mathcal{L}_{\text{GeniWorld}}(\theta) = \mathbb{E}_{t, s \sim \mathcal{U}[0, 1], z_{t+1}, \epsilon \sim \mathcal{N}(0, I)} \left[ \left\| v_\theta\left(z_{t+1}^{(s)}, s, z_{\le t} \mid z_{a, t+1}, c\right) - (z_{t+1} - \epsilon) \right\|_2^2 \right] $$

> [!NOTE]
> **Asimetría de Ruido**: Solo la observación visual latente $z_v$ se corrompe con ruido $\epsilon$ durante el entrenamiento. La acción visual $z_a$ permanece como una señal de condicionamiento determinista exacta. Esto fuerza al campo de velocidades $v_\theta$ a predecir únicamente las transiciones ambientales inducidas por el contacto físico del robot.

---

## 4. Objetivo de la Política Downstream (Action Flow Matching en $\pi_0$)

Para el control del robot, la política $\pi_0$ (VLA) se entrena a su vez mediante un objetivo de Flow Matching condicional inverso: generar el chunk de acciones futuras $A_t = [a_t, \dots, a_{t+H-1}]$ a partir de la observación $o_t$ generada por el modelo de mundo o del entorno real.

Dada la interpolación con ruido en el espacio de acciones para el tiempo de flujo $\tau \in [0, 1]$:

$$ A_t^\tau = \tau A_t + (1 - \tau)\epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

La función de pérdida de la política es:

$$ \mathcal{L}_{\text{FM Policy}}(\theta_{\text{pol}}) = \mathbb{E}_{\tau \sim \mathcal{U}[0, 1], A_t, \epsilon, o_t} \left[ \left\| v_{\theta_{\text{pol}}}(A_t^\tau, o_t) - (A_t - \epsilon) \right\|_2^2 \right] $$

---

## 5. Referencias
- [[2026_GeniWorld]]: Paper principal que implementa este objetivo.
- [[GeniWorld]]: Entidad arquitectónica.
- [[LeWM_Loss]]: Comparativa con objetivos JEPA basados en MSE y regularización Gaussiana ([[SIGReg]]).
- [[VJEPA_Loss]]: Comparativa con objetivos variacionales / probabilísticos.

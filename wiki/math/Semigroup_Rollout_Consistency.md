---
title: "Semigroup Rollout Consistency Loss"
type: "math"
tags: [math, loss, semigroup, rollout, jepa, sigreg, world-models]
---

# Semigroup Rollout Consistency Loss ($\mathcal{L}_{\text{roll}}$)

La **pérdida de consistencia de rollout bajo semigrupo** es el objetivo de entrenamiento formulado en [[2026_Semigroup-JEPA]] para superar la ceguera al horizonte temporal que sufren los modelos de mundo entrenados con *teacher forcing* a un solo paso (como [[LeWorldModel]] y [[LeWM_Loss]]).

Formaliza la noción de que las dinámicas temporales repetidas en el espacio latente deben formar un **semigrupo discreto de evolución** que sea internamente consistente y preserve las invariantes físicas a lo largo de múltiples pasos de predicción autorregresiva.

---

## 1. El Semigrupo de Evolución Latente

Sea $\mathcal{Z} = \mathbb{R}^{d_z}$ el espacio de incrustación latente producido por un encoder visual $e_\phi: \mathcal{O} \to \mathcal{Z}$. Consideremos una secuencia temporal de observaciones $\{o_t\}_{t \ge 1}$ bajo una ley física gobernada por un parámetro escalar constante $g$ (e.g., gravedad), sin acciones externas de control ($u_t = \emptyset$).

A partir de una ventana de historia de $H$ latentes consecutivos, denotamos el estado de historia como:
$$\mathbf{z}_t^{(H)} = (z_{t-H+1}, z_{t-H+2}, \dots, z_t) \in \mathcal{Z}^H$$

El predictor parametrizado $p_\theta$, condicionado por el embedding de la ley física $c_g = q_\psi(g)$, induce un operador de actualización de historia:
$$F_{\theta,g}: \mathcal{Z}^H \to \mathcal{Z}^H$$
$$F_{\theta,g}\left(z_{t-H+1:t}\right) = \left(z_{t-H+2:t},\, p_\theta(z_{t-H+1:t}, c_g)\right)$$

### Propiedad de Semigrupo Discreto
Para cualquier número de pasos de iteración $k \in \mathbb{N}_0$, definimos la $k$-ésima potencia funcional del operador:
$$S_{\theta,g}(k) \triangleq F_{\theta,g}^{\circ k}$$

La familia $\{S_{\theta,g}(k)\}_{k \in \mathbb{N}_0}$ forma un **semigrupo conmutativo/asociativo unario** que satisface:
1. **Identidad**: $S_{\theta,g}(0) = I$ (el operador identidad en $\mathcal{Z}^H$).
2. **Ley de Composición**: Para cualesquiera $k, \ell \in \mathbb{N}_0$:
   $$S_{\theta,g}(k + \ell) = S_{\theta,g}(\ell) \circ S_{\theta,g}(k) = S_{\theta,g}(k) \circ S_{\theta,g}(\ell)$$

En presencia de acciones de control variantes en el tiempo $u_t$, los operadores de actualización sucesivos ya no son idénticos sino que componen una cadena no homogénea de operadores $F_{\theta, c_t}$, preservando la estructura algebraica de composición secuencial.

---

## 2. Formulación Matemática de la Pérdida de Rollout

En lugar de evaluar únicamente el error local a un paso con latentes reales (*teacher forcing*), SG-JEPA ejecuta un rollout autorregresivo cerrado de $K$ pasos en el espacio latente.

Definimos la trayectoria autorregresiva $z_s^{\text{roll}}$ como:
$$z_s^{\text{roll}} = \begin{cases} z_s & \text{si } s \le t \quad (\text{latentes reales codificados}) \\ \hat{z}_s & \text{si } s > t \quad (\text{latentes predichos recursivamente}) \end{cases}$$

Para cada paso $k \in \{1, \dots, K\}$, el predictor genera el siguiente estado latente:
$$\hat{z}_{t+k} = p_\theta\left(z^{\text{roll}}_{t+k-H:t+k-1},\, c_{t+k-H:t+k-1}\right)$$

La pérdida de rollout normalizada con descuento exponencial está dada por:
$$\mathcal{L}_{\text{roll}} = \sum_{k=1}^K w_k \|\hat{z}_{t+k} - z_{t+k}\|_2^2$$

donde los pesos normalizados $w_k$ se calculan a partir de un factor de descuento $\gamma \in (0, 1]$:
$$w_k = \frac{\gamma^{k-1}}{\sum_{j=1}^K \gamma^{j-1}}$$

### Justificación del Descuento Exponencial ($\gamma = 0.95$)
- Si $\gamma = 1$, los pasos finales del rollout (donde los errores pueden haber crecido sustancialmente) dominan la magnitud del gradiente, desestabilizando el entrenamiento temprano.
- Si $\gamma \ll 1$, la pérdida degenera al objetivo de un solo paso de LeWM, perdiendo la sensibilidad a la deriva temporal.
- El valor óptimo $\gamma = 0.95$ proporciona un balance donde el predictor recibe señal supervisada a largo plazo sin que los errores extremos de rollout colapsen el flujo del gradiente.

---

## 3. Dinámica del Gradiente y Ausencia de Stop-Gradient

Una diferencia fundamental entre SG-JEPA y arquitecturas previas ([[2023_I-JEPA]], [[2025_V-JEPA2]]) reside en el tratamiento del **Target Encoder**:

| Característica | I-JEPA / V-JEPA | LeWorldModel (LeWM) | SG-JEPA |
| :--- | :--- | :--- | :--- |
| **Encoder Target** | Copia EMA desconectada | Mismo encoder $e_\phi$ | Mismo encoder $e_\phi$ |
| **Operador Stop-Gradient** | Sí ($\text{sg}[z_{\text{tgt}}]$) | No | **No** |
| **Horizonte de Pérdida** | 1 paso | 1 paso (Teacher-Forcing) | **$K$ pasos autorregresivos** |
| **Mecanismo Anti-Colapso** | Asimetría EMA + Masking | [[SIGReg]] | **[[SIGReg]]** |

### Flujo del Gradiente hacia la Representación
Dado que los targets $z_{t+k} = e_\phi(o_{t+k})$ provienen del mismo encoder que los contextos iniciales, el gradiente total respecto a los parámetros del encoder $\phi$ es:
$$\nabla_\phi \mathcal{L}_{\text{roll}} = \sum_{k=1}^K w_k \left( \frac{\partial \hat{z}_{t+k}}{\partial \phi} - \frac{\partial z_{t+k}}{\partial \phi} \right)^\top \left(\hat{z}_{t+k} - z_{t+k}\right)$$

La derivada $\frac{\partial \hat{z}_{t+k}}{\partial \phi}$ incorpora tanto la dependencia de $\hat{z}_{t+k}$ respecto a los primeros $H$ estados de contexto $z_{t-H+1:t} = e_\phi(o_{t-H+1:t})$ como su propagación a través de las llamadas autorregresivas intermedias del predictor.

Esto genera una presión directa sobre el encoder: **aprender representaciones donde las dinámicas físicas sean proyectables mediante operadores que se componen sin perder información esencial**.

---

## 4. Función de Pérdida Total Unificada

Para garantizar que el encoder no colapse hacia una solución trivial constante ($z_t = \mathbf{c} \implies \mathcal{L}_{\text{roll}} = 0$), se acopla el regularizador isotrópico gaussiano de Balestriero y LeCun ([[SIGReg]]):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{roll}} + \lambda_{\text{sig}} \mathcal{L}_{\text{SIGReg}}(Z)$$

donde $Z \in \mathbb{R}^{B \times d_z}$ es la matriz de latentes codificados en el batch, y $\mathcal{L}_{\text{SIGReg}}$ proyecta $Z$ sobre direcciones aleatorias uniformes $v \sim \mathbb{S}^{d_z-1}$ penalizando la discrepancia de la **función característica empírica** frente a $e^{-t^2/2}$, la CF de una Gaussiana estándar $\mathcal{N}(0, I)$. Se usan 1024 proyecciones y 17 knots (SG-JEPA App. C; Eq. 160, App. H.8). El paper no nombra el test, pero la forma coincide con Epps–Pulley ([[SIGReg]]). Hay un matiz: no se usa stop-gradient ni EMA. Los targets salen del mismo encoder entrenable, y el propio paper advierte que su análisis local "neither proves noncollapse nor attributes the dynamics difference to SIGReg" (App. H.8).

---

## 5. Referencias Cruzadas
- **Teoría Analítica del Error**: [[Latent_Dynamics_Consistency]]
- **Regularizador Base**: [[SIGReg]]
- **Arquitectura**: [[2026_Semigroup-JEPA]]
- **Paper**: [[2026_Semigroup-JEPA]]
- **Entidad**: [[SG-JEPA]]

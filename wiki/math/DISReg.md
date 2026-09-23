---
title: "DISReg: Difference Image and Single Image Embedding Regularization"
tags: [math, loss, disreg, sigreg, jepa, anti-collapse, slow-features, inverse-dynamics]
---

# DISReg (Difference Image and Single Image Embedding Regularization)

**DISReg** es una función de regularización y aprendizaje autosupervisado introducida en [[2026_MotionJEPA]] para prevenir el **colapso temporal de características** (*temporal feature collapse*) en modelos de mundo basados en incrustaciones conjuntas ([[LeWorldModel]], [[LeJEPA]], [[2025_V-JEPA2]]).

A diferencia de los regularizadores clásicos que actúan únicamente sobre la distribución estática de los embeddings latentes (e.g., [[SIGReg]], VICReg), DISReg acopla la dinámica de cambio visual en el espacio de diferencias con la predicción latente, garantizando que el encoder preserve tanto la información estática del entorno como las transiciones rápidas de movimiento.

---

## 1. Motivación Matemática: El Sesgo de Características Lentas

Consideremos una secuencia de observaciones $o_t \in \mathbb{R}^{3 \times H \times W}$ descompuesta aditivamente en un componente estático de fondo $s_t = s$ (invariante o de variación ultra-lenta) y un componente dinámico de primer plano $m_t$ (e.g. actuador, proyectil):
$$o_t = s + m_t$$

Un encoder JEPA estándar $z_t = e_\theta(o_t) \in \mathbb{R}^{D_z}$ entrenado con pérdida de predicción forward:
$$\mathcal{L}_{\text{fwd}} = \|p_\phi(z_t, a_t) - z_{t+1}\|_2^2$$
junto con un regularizador estático anti-colapso $\mathcal{L}_{\text{anti}} = \text{SIGReg}(Z)$, minimiza la pérdida mucho más fácilmente si $e_\theta$ codifica predominantemente $s$:
$$\frac{\partial \|z_{t+1} - \hat{z}_{t+1}\|_2^2}{\partial z_t} \to 0 \quad \text{si } z_t \approx f(s)$$
dado que predecir una señal casi constante trivializa la optimización de $p_\phi$. Esto produce **supresión de características dinámicas**: $m_t$ se pierde en la proyección a $z_t$, provocando que el modelo de mundo falle en el control de agentes y la predicción de colisiones.

---

## 2. Formulación de DISReg

DISReg introduce dos transformaciones complementarias:

### A. Extracción en el Espacio de Diferencias
Se calcula la imagen de diferencia temporal directa:
$$o^{\text{diff}}_t \triangleq o_{t+1} - o_t$$
Nótese que:
$$o^{\text{diff}}_t = (s + m_{t+1}) - (s + m_t) = m_{t+1} - m_t$$
El componente estático $s$ se anula idénticamente, aislando estrictamente la información cinemática y de cambio visual sin necesidad de segmentación ni etiquetas semánticas.

### B. Módulos de Diferencia Latente
1. **Difference Encoder ($\text{DiffEnc}_\alpha$)**:
   $$d_t = \text{DiffEnc}_\alpha(o^{\text{diff}}_t) \in \mathbb{R}^{D_d}$$
2. **Difference Predictor ($\text{DiffPred}_\beta$)**:
   $$\hat{d}_t = \text{DiffPred}_\beta(z_t, z_{t+1}) \in \mathbb{R}^{D_d}$$

---

## 3. Función de Pérdida y Desglose de Componentes

La pérdida DISReg se define como una combinación lineal de tres términos:
$$\mathcal{L}_{\text{DISReg}} \triangleq \lambda_z L_z + \lambda_d L_d + \lambda_{\text{pred}} L_{\text{pred}}$$

### 1. Término Estático ($L_z$)
$$L_z \triangleq \text{SIGReg}(Z)$$
Aplica la prueba de Cramér-von Mises proyectada sobre direcciones aleatorias uniformes $v \sim \mathbb{S}^{D_z-1}$ para ajustar la distribución empírica de los embeddings de estado $Z \in \mathbb{R}^{B \times D_z}$ a una distribución normal estándar isotrópica $\mathcal{N}(0, I_{D_z})$. Preserva la diversidad informacional global del estado.

### 2. Término Dinámico ($L_d$)
$$L_d \triangleq \text{SIGReg}(D)$$
Aplica SIGReg sobre el batch de embeddings de diferencia $D \in \mathbb{R}^{B \times D_d}$, garantizando que $\text{DiffEnc}_\alpha$ no colapse a una representación constante y mantenga un espacio de diferencias de rango completo.

### 3. Término de Predicción de Diferencias ($L_{\text{pred}}$)
$$L_{\text{pred}} \triangleq \frac{1}{B} \sum_{i=1}^B \|d_t^{(i)} - \hat{d}_t^{(i)}\|_2^2 = \frac{1}{B} \sum_{i=1}^B \|d_t^{(i)} - \text{DiffPred}_\beta(z_t^{(i)}, z_{t+1}^{(i)})\|_2^2$$

### Objetivo Total de MotionJEPA
$$\mathcal{L}_{\text{MotionJEPA}} = \mathcal{L}_{\text{DISReg}} + \|\hat{z}_{t+1} - z_{t+1}\|_2^2$$

---

## 4. Análisis del Flujo de Gradiente y No-Restricción Geométrica

El gradiente de $L_{\text{pred}}$ respecto al embedding de estado $z_t$ es:
$$\nabla_{z_t} L_{\text{pred}} = -2 \left( \frac{\partial \text{DiffPred}_\beta(z_t, z_{t+1})}{\partial z_t} \right)^\top (d_t - \hat{d}_t)$$

> [!NOTE]
> **Propiedad Clave**: A diferencia de regularizadores como LeWM-Time o LeNEPA, que imponen restricciones de forma o distribución gaussianas directamente a lo largo del tiempo sobre $z$ (lo que distorsiona la física no gaussiana de movimientos periódicos o colisiones), $L_{\text{pred}}$ **no impone restricciones directas sobre la distribución geométrica de $z$**. La presión sobre $z$ es puramente funcional: $z_t$ y $z_{t+1}$ deben contener bits informacionales suficientes para que la red MLP $\text{DiffPred}_\beta$ pueda sintetizar el embedding de cambio $d_t$.

---

## 5. Comparativa Teórica de Métodos Anti-Colapso Temporal

| Método | Señal Anti-Colapso Temporal | Requiere Acciones ($a_t$) | Captura Dinámica Pasiva | Distorsiona Geometría Temporal de $z$ |
| :--- | :--- | :--- | :--- | :--- |
| **LeWorldModel (LeWM)** | Ninguna (solo batch SIGReg) | No | No (colapsa) | No |
| **LeWM-Time / LeNEPA** | SIGReg en dimensión tiempo | No | Parcial | **Sí** (fuerza $z(t)$ a gaussiana) |
| **SMWM** | $\text{Inv}(z_t, z_{t+1}) \to a_t$ | **Sí** | **No** | No |
| **Delta-JEPA** | $\text{Inv}(\Delta z_t) \to a_t$ | **Sí** | **No** | No |
| **DISReg (MotionJEPA)** | $\text{DiffPred}(z_t, z_{t+1}) \to d_t$ | **No** | **Sí** | **No** |

---

## 6. Hiperparámetros Óptimos
- $\lambda_z = 0.25$: Ponderación del término estático de estado.
- $\lambda_d = 2.0$: Ponderación del anti-colapso en el espacio de diferencias.
- $\lambda_{\text{pred}} = 0.5$: Ponderación de la predicción de diferencias.

---

## 7. Referencias Cruzadas
- **Paper**: [[2026_MotionJEPA]]
- **Arquitectura**: [[2026_MotionJEPA]]
- **Regularizador Base**: [[SIGReg]]
- **Pérdida Base**: [[LeWM_Loss]]
- **Consistencia Temporal**: [[Semigroup_Rollout_Consistency]]

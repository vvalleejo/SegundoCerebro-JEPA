---
title: "Latent Dynamics Consistency & Linear Feature Theory"
tags: [math, theory, dynamics, semigroup, jepa, consistency, generalization, linear-feature-model]
---

# Latent Dynamics Consistency & Linear Feature Theory

Este documento detalla el marco analítico desarrollado en [[2026_Semigroup-JEPA]] (Apéndice H) para fundamentar rigurosamente:
1. Las condiciones bajo las cuales un modelo de mundo latente generaliza fuera de distribución (OOD) ante variaciones en las leyes físicas (ej. gravedad $g$).
2. Cómo se acumulan y amplifican los errores locales de predicción bajo composición en semigrupos de evolución temporal.
3. Por qué la optimización *teacher-forced* a un solo paso falla al seleccionar representaciones cerradas dinámicamente, provocando una **inversión de ranking** frente a la evaluación a largo plazo.

---

## 1. El Modelo de Características Lineales (Linear Feature Model)

Consideremos un espacio de características latentes completas (pero no necesariamente observables directamente) $\phi_t \in \mathbb{R}^{d_\phi}$ cuyas dinámicas microscópicas siguen una ecuación en diferencias lineal estocástica parametrizada por una ley física $g$:
$$\phi_{t+1} = T(g) \phi_t + \xi_{t+1}, \qquad \mathbb{E}[\xi_{t+1} \mid \phi_t, g] = 0$$
donde $T(g) \in \mathbb{R}^{d_\phi \times d_\phi}$ es la matriz de transición y $\xi_{t+1}$ es ruido de media cero con covarianza acotada.

### Proyección Latente del Encoder
El encoder comprime el estado $\phi_t$ a un espacio latente de menor dimensión $z_t \in \mathbb{R}^{d_z}$ ($d_z \le d_\phi$) mediante una proyección ortonormal por filas $W \in \mathbb{R}^{d_z \times d_\phi}$:
$$z_t = W \phi_t, \qquad W W^\top = I_{d_z}$$

### Operador Proyectado y Matriz de Clausura Predictiva
Definimos:
- **Operador proyectado ideal en el subespacio latente**:
  $$A_W(g) \triangleq W T(g) W^\top \in \mathbb{R}^{d_z \times d_z}$$
- **Matriz de defecto de clausura (Closure Matrix)**:
  $$C_W(g) \triangleq W T(g) - A_W(g) W \in \mathbb{R}^{d_z \times d_\phi}$$

Nótese que multiplicando a la derecha por $W^\top$:
$$C_W(g) W^\top = W T(g) W^\top - A_W(g) W W^\top = A_W(g) - A_W(g) = 0$$
Por tanto, $C_W(g)$ actúa exclusivamente sobre el subespacio ortogonal complementario al espacio latente capturado por $W$, es decir, sobre las componentes descartadas $(I - W^\top W) \phi_t$.

---

## 2. Defecto Medio Condicional y Clausura Predictiva

Dado un predictor entrenado $\hat{A}(g) \in \mathbb{R}^{d_z \times d_z}$, cuando se aplica de manera *teacher-forced* sobre el estado latente real $z_t$, su predicción es $\hat{z}_{t+1}^{\text{TF}} = \hat{A}(g) z_t$.

### Definición: Defecto Medio Condicional ($\delta_t(g)$)
$$\delta_t(g) \triangleq \mathbb{E}\left[ z_{t+1} - \hat{z}_{t+1}^{\text{TF}} \,\middle|\, \phi_t, g \right] = W T(g) \phi_t - \hat{A}(g) W \phi_t$$

Descomponiendo $W T(g) = A_W(g) W + C_W(g)$:
$$\delta_t(g) = \underbrace{C_W(g) \phi_t}_{\text{Error de Clausura}} + \underbrace{[A_W(g) - \hat{A}(g)] z_t}_{\text{Error del Predictor}}$$

El error observado en un solo paso es:
$$z_{t+1} - \hat{z}_{t+1}^{\text{TF}} = \delta_t(g) + W \xi_{t+1}$$

### Clausura Predictiva ($C_W(g) = 0$)
Se dice que la representación $W$ es **predictivamente cerrada** bajo la ley $g$ si y solo si:
$$C_W(g) = 0 \iff W T(g) = A_W(g) W$$
Algebraicamente, esto significa que el subespacio $\operatorname{span}(W^\top)$ es un subespacio invariante a la izquierda para $T(g)$. Físicamente, implica que el estado latente $z_t$ contiene toda la información necesaria para determinar la esperanza condicional de $z_{t+1}$, sin ninguna fuga hacia variables ocultas de $\phi_t$.

---

## 3. Cota de Generalización a Leyes Físicas OOD (Law Coverage Bound)

¿Cómo afecta evaluar el modelo en una gravedad no vista $g^\star \notin \text{supp}(P_{\text{tr}})$?

### Descomposición en Base de Leyes Físicas
Supongamos que el operador de transición depende linealmente de una base finita de funciones de la ley $\psi(g) = [\psi_1(g), \dots, \psi_m(g)]^\top \in \mathbb{R}^m$:
$$T(g) = \sum_{k=1}^m \psi_k(g) T_k, \qquad \hat{A}(g) = \sum_{k=1}^m \psi_k(g) \hat{A}_k$$

Definimos la matriz de segundos momentos sobre la distribución de entrenamiento $P_{\text{tr}}$:
$$M_\psi \triangleq \mathbb{E}_{g \sim P_{\text{tr}}}\left[ \psi(g) \psi(g)^\top \right] \succ 0$$

### Factor de Cobertura de la Ley ($L_{\text{law}}$)
Para cualquier parámetro de evaluación $g^\star$, definimos el factor de cobertura como la distancia de Mahalanobis en el espacio de la ley:
$$L_{\text{law}}(g^\star) \triangleq \psi(g^\star)^\top M_\psi^{-1} \psi(g^\star)$$

### Teorema H.4 (Cota de Error Local en Gravedad No Vista)
Bajo los supuestos de base compartida, el defecto medio en $g^\star$ está acotado por:
$$\|\delta_t(g^\star)\|_2 \le \sqrt{L_{\text{law}}(g^\star)} \left( B_z \epsilon_{\text{op}} + B_\phi \epsilon_{\text{cl}} \right)$$
donde:
- $B_z = \sup_t \|z_t\|_2$ y $B_\phi = \sup_t \|\phi_t\|_2$ acotan las normas de los estados.
- $\epsilon_{\text{op}} = \sqrt{\mathbb{E}_{g \sim P_{\text{tr}}}\|A_W(g) - \hat{A}(g)\|_F^2}$ es el error cuadrático medio del predictor en entrenamiento.
- $\epsilon_{\text{cl}} = \sqrt{\mathbb{E}_{g \sim P_{\text{tr}}}\|C_W(g)\|_F^2}$ es el defecto de clausura cuadrático medio del encoder en entrenamiento.

### Corolario H.5: Dinámicas Balísticas Afines
Para caída libre y tiro parabólico, la aceleración gravitatoria entra de forma estrictamente afín en la integración temporal ($v_{t+1} = v_t + g\Delta$, $x_{t+1} = x_t + v_t\Delta + \frac{1}{2}g\Delta^2$), por lo que $\psi(g) = [1, g]^\top$.

Si la distribución de entrenamiento tiene media $\mu_{\text{tr}}$ y varianza $\sigma_{\text{tr}}^2$, la matriz de momentos es:
$$M_\psi = \begin{bmatrix} 1 & \mu_{\text{tr}} \\ \mu_{\text{tr}} & \mu_{\text{tr}}^2 + \sigma_{\text{tr}}^2 \end{bmatrix} \implies M_\psi^{-1} = \begin{bmatrix} 1 + \frac{\mu_{\text{tr}}^2}{\sigma_{\text{tr}}^2} & -\frac{\mu_{\text{tr}}}{\sigma_{\text{tr}}^2} \\ -\frac{\mu_{\text{tr}}}{\sigma_{\text{tr}}^2} & \frac{1}{\sigma_{\text{tr}}^2} \end{bmatrix}$$

El factor de cobertura toma la forma cuadrática exacta:
$$L_{\text{law}}(g^\star) = 1 + \frac{(g^\star - \mu_{\text{tr}})^2}{\sigma_{\text{tr}}^2}$$

> [!NOTE]
> **Implicación Teórica Principal**: Condicionar por el parámetro físico $g$ no basta por sí solo para garantizar extrapolación. El error OOD escala proporcionalmente a la distancia estandarizada $|g^\star - \mu_{\text{tr}}| / \sigma_{\text{tr}}$ modulado por la suma de la precisión del predictor $\epsilon_{\text{op}}$ y, crucialmente, la clausura de la representación $\epsilon_{\text{cl}}$.

---

## 4. Acumulación Telescópica del Error en Semigrupos

A gravedad constante $g$, los operadores de evolución verdadera y aprendida generan los semigrupos discretos $S_g(h) = T(g)^h$ y $\hat{S}_g(h) = \hat{A}(g)^h$.

Partiendo del mismo latente inicial $z_0 = \hat{z}_0 = W \phi_0$, el error de rollout libre tras $h$ pasos recursivos, $e_h \triangleq z_h - \hat{z}_h$, satisface la identidad telescópica exacta (Teorema H.10):
$$e_h = \sum_{j=0}^{h-1} \hat{A}(g)^{h-1-j} \left[ \delta_j(g) + W \xi_{j+1} \right]$$

### Análisis de Amplificación Recursiva
Cada defecto local $\delta_j(g)$ introducido en el paso $j$ no permanece aislado, sino que es propagado por los $h - 1 - j$ operadores restantes del semigrupo $\hat{A}(g)$.
- Si $\hat{A}(g)$ tiene valores propios con magnitud cercana o superior a 1 en las direcciones alineadas con $\delta_j(g)$, el error crece polinomial o exponencialmente con el horizonte $h$.
- Por tanto, dos encoders que tengan idéntico error local a un paso $\|\delta_t\|_2$ pueden exhibir divergencias astronómicamente dispares a horizonte $h = 44$, dependiendo de si los defectos caen en el subespacio contractivo o expansivo de $\hat{A}(g)$.

---

## 5. El Teorema de Inversión de Ranking (Ranking Reversal Theorem)

¿Por qué el entrenamiento *teacher-forcing* a un paso de LeWM es incapaz de encontrar representaciones óptimas para rollouts largos?

### Defecto de Entrelazamiento de Semigrupo (Intertwining Defect)
Definimos el operador de defecto a horizonte $h$:
$$D_h(W, \hat{A}) \triangleq W T(g)^h - \hat{A}(g)^h W$$

Para $h = 1$, $D_1(W, \hat{A}) = W T(g) - \hat{A}(g) W = C_W(g) + (A_W(g) - \hat{A}(g))W$.

### Teorema H.15 (Inversión Estable de Ranking)
Existe una dinámica lineal en $\mathbb{R}^{4}$ con matrices $T$ y operadores de proyección $W_1, W_2 \in \mathbb{R}^{2 \times 4}$ y predictores correspondientes $\hat{A}_1, \hat{A}_2$ tales que:
$$\|D_1(W_1, \hat{A}_1)\|_F^2 < \|D_1(W_2, \hat{A}_2)\|_F^2$$
pero a partir de un horizonte finito $h \ge 2$:
$$\|D_h(W_1, \hat{A}_1)\|_F^2 \gg \|D_h(W_2, \hat{A}_2)\|_F^2$$

### Significado Matemático y Conceptual
Un optimizador guiado por una pérdida de 1 paso (como LeWM) seleccionará inexorablemente la representación $W_1$ porque su error inmediato es menor. Sin embargo, $W_1$ descarta variables latentes lentas o armónicas que desestabilizan la composición a largo plazo.

Por el contrario, al entrenar con la **pérdida de rollout de $K$ pasos $\mathcal{L}_{\text{roll}}$ retropropagada directamente en el encoder $W$**, el gradiente penaliza explícitamente los términos $\hat{A}^{h-1-j} C_W(g)$, forzando al encoder a seleccionar $W_2$, el cual anula la clausura $C_W(g) \approx 0$ y asegura estabilidad espectral bajo el semigrupo de evolución.

---

## 6. Referencias Cruzadas
- **Formulación de la Pérdida**: [[Semigroup_Rollout_Consistency]]
- **Paper**: [[2026_Semigroup-JEPA]]
- **Arquitectura**: [[2026_Semigroup-JEPA]]
- **Línea de Identificabilidad**: [[Linear_Identifiability]]
- **Regularizador**: [[SIGReg]]

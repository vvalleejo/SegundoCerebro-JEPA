---
title: "AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving"
authors: [Haoran Zhu, Wancong Zhang, Yann LeCun, Anna Choromanska]
year: 2026
venue: "arXiv preprint"
arxiv: "2609.34085"
source_pdf: "raw/AD-E2E-JEPA A JOINT-EMBEDDING PREDICTIVE ARCHITECTURE FOR END-TO-END AUTONOMOUS DRIVING.pdf"
repo: "https://github.com/HaoranZhuExplorer/AD-E2E-JEPA"
type: paper
family: jepa
modality: [video, control]
anti_collapse: [frozen-encoder, sg, sigreg]
predictor: true
planner: [vocab-search]
tags: [paper, jepa, world-models, autonomous-driving, e2ead, sigreg, dinov3, navsim, zero-shot-planning]
---

# AD-E2E-JEPA: JEPA para conducción autónoma end-to-end

> [!abstract] TL;DR
> World model JEPA acción-condicionado para conducción (solo cámara frontal) sobre **DINOv3 ViT-L congelado**, con un **patch projector** entrenable (2 convoluciones de stride $2\times2$) que comprime los parches de $16\times32\times1024$ a $4\times8\times256$: 16× menos tokens y 4× menos dimensión (Fig. 2). Se entrena con MSE contra target con stop-gradient + **[[SIGReg]] aplicado parche a parche** (Eq. 9–11). Sin entrenar ninguna política, planifica **eligiendo de un vocabulario de trayectorias** la que lleva el rollout latente más cerca del embedding de un **frame futuro real usado como objetivo** (Eq. 13–14). Rinde como [[DINO-WM]] y [[JEPA-WM]] (EPDMS 76.6 frente a 68.3 / 74.2 en 100 escenas), pero planifica en 0.8 s frente a 91.8 / 101.0 s (Tab. 2). El proyector preentrenado también mejora un modelo de imitation learning: EPDMS 80.2 → 85.4 (Tab. 3).

> [!warning] Procedencia
> - **Autoría:** Haoran Zhu, Wancong Zhang, Yann LeCun, Anna Choromanska. Afiliaciones: New York University; Zhang y LeCun también AMI Labs (PDF p. 1).
> - **Venue:** arXiv:2609.34085v1 [cs.RO], 28 Sep 2026 (PDF p. 1). Preprint sin revisión por pares indicada.
> - **Tipo:** empírico (sin resultados teóricos).
> - **Sin señales de alerta graves:** autoría coherente con su línea (Zhu y Choromanska firman AD-L-JEPA y la línea JEPA-LiDAR; Zhang es coautor de [[PLDM]] y de *Hierarchical planning with latent world models*). Hay experimentos con benchmark público (NAVSIMv2) y código anunciado en GitHub.
> - **Señales menores:** una sola ejecución por configuración, sin desviaciones ni semillas; no hay ablación de SIGReg (¿qué pasa con $\lambda=0$?); baseline LeWM con ViT-L desde cero y batch 8 (Tab. 1), lejos de su configuración original. Ver Limitaciones.

---

## Resumen ejecutivo

**Problema.** Los sistemas de conducción end-to-end (E2EAD) dependen casi siempre de imitation learning: imitan trayectorias humanas sin modelar la dinámica del entorno (PDF §1). Un world model JEPA acción-condicionado permitiría evaluar trayectorias candidatas antes de ejecutarlas. Sin embargo, los world models JEPA existentes presentan un *trade-off* (PDF §1, Tab. 2):
- [[LeWorldModel|LeWM]] (CLS global) es rápido (0.7 s/escena) pero planifica mal (EPDMS 48.3, FDE 12.4 m).
- [[DINO-WM]] y [[JEPA-WM]] (parches densos de DINOv3) planifican bien pero tardan más de 90 s por escena.

**Propuesta.** Partir de la mejor configuración de JEPA-WM (DINOv3 ViT-L + predictor AdaLN con RoPE, rollout opcional) e insertar un proyector convolucional compartido entre las ramas de contexto y de target, inspirado en el proyector de [[wiki/papers/2026_Temporal_Straightening|Temporal Straightening]] (allí con stride $1\times1$). Como el proyector se entrena, el target ya no está congelado y aparece el riesgo de colapso. Para evitarlo usan stop-gradient en el target + SIGReg (PDF §3.2.2).

**Contribuciones declaradas** (PDF §1):
1. Primera adaptación de JEPA a E2EAD con planificación *goal-conditioned zero-shot*, con métricas más allá del éxito: EPDMS, tiempo, FDE/$\Delta x$/$\Delta y$/$\Delta\theta$ y *hit rate*.
2. Mostrar el trade-off eficiencia/precisión de los world models JEPA existentes.
3. AD-E2E-JEPA: reducción 100× del tiempo de planificación sin perder rendimiento.
4. Transferencia del proyector preentrenado a imitation learning (+5.2 EPDMS).

---

## Arquitectura

Detalle completo, figuras y shapes en [[wiki/architecture/2026_AD-E2E-JEPA_Arch|arquitectura AD-E2E-JEPA]].

- **Encoder:** DINOv3 ViT-L **congelado**, $D_{\text{enc}}=1024$; entrada $3\times256\times512$ ⇒ $16\times32$ parches (Fig. 2, App. A.5).
- **Patch projector** $\mathrm{Proj}$: 2 × Conv2D stride $2\times2$, entrenable y **compartido** entre historia y futuro: $16\times32\times1024\to4\times8\times256$ (Eq. 7; Fig. 2 a.1).
- **Encoder de acción** $E_a$: capa lineal sobre $a_t=[\Delta x,\Delta y,\Delta\theta]^\top$, el cambio de pose relativo entre frames (Eq. 3).
- **Predictor** $\mathrm{Pred}'$: transformer de parches con condicionamiento AdaLN + RoPE (herencia de JEPA-WM), un paso adelante sobre la ventana (Eq. 8).
- **Anti-colapso:** encoder congelado (DINOv3) + stop-gradient en el target proyectado + SIGReg parche a parche. Sin EMA.

---

## Formulación

**Notación del paper y equivalencias con la canónica de la bóveda:** $K$ = longitud de la historia (aquí $K+1=4$ frames); **no** son las direcciones de SIGReg, que el paper llama $M$ (canónico: $K$). $F$ = horizonte futuro ($F=8$, canónico $H$). $B$ = batch. $D$ = dimensión del embedding ($D_{\text{proj}}=256$). $N=K+2$ = longitud de la ventana temporal en Eq. 10, **no** el nº de tokens. $T(\cdot)$ = test de Epps–Pulley, **no** el nº de nodos de cuadratura. $H, W$ = alto y ancho de la imagen; $H'\times W'=4\times8$ = rejilla proyectada.

**Tarea** (Eq. 1–2): con historia $(I_{t-K:t},P_{t-K:t})$, $I_j\in\mathbb R^{3\times H\times W}$, $P_j=[x_j,y_j,\theta_j]^\top\in\mathbb R^3$ relativas a la pose actual ($P_t=0$), predecir $\hat P_{t+1:t+F}$.

**Baseline JEPA-WM adaptado** (Eq. 4–6): $s=\mathrm{Enc}(I)$, $\hat s_{t-K+1:t+1}=\mathrm{Pred}(s_{t-K:t},E_a(a_{t-K:t}))$, $\mathcal L_{\text{pred}}=\mathrm{MSE}(\hat s,s)$. No usan término anti-colapso porque el target sale de DINOv3 congelado.

**AD-E2E-JEPA (un paso)** (Eq. 7–11):

$$z_{t-K:t+1}=\mathrm{Proj}(s_{t-K:t+1}),\qquad \hat z_{t-K+1:t+1}=\mathrm{Pred}'(z_{t-K:t},E_a(a_{t-K:t}))$$

$$\mathcal L=\underbrace{\mathrm{MSE}\big(\hat z_{t-K+1:t+1},\,\mathrm{sg}(z_{t-K+1:t+1})\big)}_{\mathcal L^{\text{proj}}_{\text{pred}}}+\lambda\,\mathcal L^{t-K:t+1}_{\text{SIGReg}}$$

Forma **aditiva**. $\lambda=0.09$ por defecto; 0.025 en trainval sin rollout (Tab. 1; §4.2).

**SIGReg parche a parche** (Eq. 10): el test se aplica por separado a cada posición espacio-temporal $l$ sobre el batch, y se promedia:

$$\mathcal L^{t-K:t+1}_{\text{SIGReg}}=\frac{1}{NH'W'M}\sum_{l=1}^{NH'W'}\sum_{m=1}^{M}T\Big(\big\{\langle z_{l,b},u^{(m)}\rangle\big\}_{b=1}^{B}\Big),\qquad u^{(m)}\in\mathbb S^{D-1}$$

con $z_{l,b}\in\mathbb R^{D}$ el embedding del parche $l$ de la muestra $b$. Configuración: 17 nodos en $[0,3]$, $M=1024$ (App. A.1). LeWM, en cambio, lo aplica al CLS global en cada paso temporal y promedia en el tiempo (PDF p. 5). Página canónica: [[SIGReg]] (§3, fila AD-E2E-JEPA).

> [!important] Lectura del índice $l$
> Eq. 10 tiene la forma $T(\{\cdot\}_{b=1}^B)$ por posición: el test es **marginal por parche**, no conjunto sobre todos los parches. Impone que cada parche, visto a lo largo del batch, sea $\mathcal N(0,I_{256})$, pero no penaliza la dependencia entre parches distintos.

**Rollout opcional** (Eq. 12; App. A.2): $\mathcal L_{\text{multi}}=\frac{\mathcal L_{\text{TF}}+\mathcal L_2+\dots+\mathcal L_F}{F}+\lambda\mathcal L^{t-K:t+F}_{\text{SIGReg}}$, con teacher forcing sobre todo el horizonte y rollout autorregresivo con TBPTT. Página: [[Teacher_Forcing_Rollout_Loss]].

**Planificación** (Eq. 13–14): con un vocabulario $\mathcal V=\{P^i_{t:t+F}\}_{i=1}^{8192}$ (anclas de VADv2, Chen et al. 2024b), ordenado por ángulo y submuestreado uniformemente ($|\mathcal V_{\text{sampled}}|=256$ por defecto):

$$C_i=\big\|z_{t+F}-\hat z^{\,i}_{t+F}\big\|_2^2,\qquad i^*=\arg\min_{i}C_i,\qquad P^*_{t:t+F}=P^{i^*}_{t:t+F}$$

$z_{t+F}=\mathrm{Proj}(\mathrm{Enc}(I_{t+F}))$ es el embedding del **frame futuro real** (*oracle goal*, Fig. 2b) y $\hat z^{\,i}_{t+F}$ es el rollout autorregresivo de $F$ pasos bajo las acciones relativas de la trayectoria $i$. No hay optimización iterativa: es una búsqueda exhaustiva sobre el vocabulario, evaluada en paralelo en GPU (App. A.3). La norma es la de Frobenius sobre $\mathbb R^{32\times256}$.

**Métricas** (App. A.3):
- $\text{EPDMS}_i=\text{NC}_i\cdot\text{DAC}_i\cdot\text{DDC}_i\cdot\text{TLC}_i\cdot\dfrac{5\,\text{EP}_i+5\,\text{TTC}_i+2\,\text{LK}_i+2\,\text{HC}_i+2b_i\,\text{EC}_i}{14+2b_i}$ (Eq. 25), con $b_i\in\{0,1\}$ la disponibilidad de escena vecina para EC (Eq. 24).
- $\text{EPDMS}^\dagger_i$ es solo el factor ponderado, sin los términos multiplicativos de seguridad (Eq. 26).
- FDE: distancia euclídea media entre la posición final elegida y la real; $\Delta x$, $\Delta y$, $\Delta\theta$: errores absolutos medios.
- Hit rate: se **añade la trayectoria real** al conjunto de candidatas, $h^{@k}_n=\mathbb I[\mathrm{rank}(C^{gt}_n)\le k]$, $\text{HitRate@}k=\frac1Q\sum_n h^{@k}_n$, $k\in\{1,5\}$ (Eq. 27–28), adaptado del código de DrivoR.

---

## Resultados clave

**Datos:** NAVSIM / NAVSIMv2. *navtrain* = 10 h a 2 Hz (comparación justa entre métodos); *trainval* = 70 h (solo AD-E2E-JEPA). Historia $K+1=4$ frames (2 s), futuro $F=8$ (4 s) (PDF §4.1–4.2).

**Entrenamiento** (Tab. 1): AdamW, 30 épocas, 1 época de warmup + coseno; lr escalado con $\sqrt{\text{batch}}$.

| Variante | Split | GPUs | Batch | lr | $\lambda$ | Tiempo |
|---|---|---|---|---|---|---|
| LeWM | navtrain | 4×A100 | 8 | $10^{-4}$ | 0.09 | 1 d |
| DINO-WM | navtrain | 4×A100 | 64 | $10^{-4}$ | – | 11 h |
| JEPA-WM | navtrain | 4×A100 | 64 | $10^{-4}$ | – | 13 h |
| AD-E2E-JEPA | navtrain | 1×A100 | 128 | $10^{-4}$ | 0.09 | 20 h |
| + rollout | navtrain | 1×A100 | 128 | $10^{-4}$ | 0.09 | 1 d 22 h |
| AD-E2E-JEPA | trainval | 4×A100 | 512 | $2\times10^{-4}$ | 0.025 | 2 d 5 h |
| + rollout | trainval | 4×A100 | 256 | $1.4\times10^{-4}$ | 0.09 | 4 d 2 h |

**Planificación zero-shot, 100 escenas de test, 256 candidatas** (Tab. 2; EC excluido porque el subconjunto no tiene escenas contiguas):

| Método | Split | EPDMS↑ | EPDMS†↑ | Tiempo (s)↓ | FDE (m)↓ | $\Delta\theta$ (°)↓ | Hit top-1/5 (%)↑ |
|---|---|---|---|---|---|---|---|
| LeWM | navtrain | 48.3 | 73.9 | 0.7 | 12.4 | 13.6 | 6/18 |
| DINO-WM | navtrain | 68.3 | 91.4 | 91.8 | 3.9 | 5.9 | 40/73 |
| JEPA-WM | navtrain | 74.2 | 90.9 | 101.0 | 4.0 | 5.2 | 45/75 |
| AD-E2E-JEPA | navtrain | **76.6** | **92.4** | 0.8 | 4.2 | 4.7 | 27/59 |
| + rollout | navtrain | 70.4 | 92.2 | 0.8 | 3.5 | 4.1 | 34/67 |
| AD-E2E-JEPA | trainval | 72.1 | 89.8 | 0.8 | 4.7 | 3.4 | 33/55 |
| + rollout | trainval | 72.3 | 91.9 | 0.8 | **3.2** | 3.6 | **47/78** |

- Aceleración frente a DINO-WM / JEPA-WM: $91.8/0.8\approx115\times$ y $101.0/0.8\approx126\times$ (cálculo de la bóveda); el paper redondea a "100×".
- Con navtrain, AD-E2E-JEPA tiene el mejor EPDMS, pero peor FDE y hit rate que DINO-WM y JEPA-WM (PDF p. 9).

**Test completo (12 146 escenas)** (Tab. 2, Tab. 5):

| Método | Split | Candidatas | EPDMS | EPDMS† | Tiempo (s) | FDE (m) | $\Delta\theta$ (°) | Hit top-1/5 (%) |
|---|---|---|---|---|---|---|---|---|
| LeWM | navtrain | 256 | 39.8 | 66.7 | 0.7 | 14.6 | 17.2 | 5.7/11.3 |
| AD-E2E-JEPA | navtrain | 256 | 63.5 | 80.1 | 0.8 | 6.3 | 6.3 | 32.9/65.3 |
| + rollout | navtrain | 256 | 64.9 | 83.1 | 0.8 | 4.5 | 4.1 | 42.6/71.0 |
| AD-E2E-JEPA | trainval | 256 | 63.2 | 80.1 | 0.8 | 6.2 | 4.5 | 31.0/64.8 |
| + rollout | trainval | 256 | 67.3 | 84.1 | 0.8 | 4.0 | 3.5 | 53.8/82.7 |
| + rollout | trainval | 512 | 69.2 | 85.0 | 1.4 | 3.6 | 2.9 | 45.2/73.9 |
| + rollout | trainval | 1024 | 70.5 | 85.5 | 2.5 | 3.2 | 2.5 | 36.3/64.3 |
| + rollout | trainval | 2048 | 71.5 | 86.0 | 4.7 | 3.0 | 2.2 | 27.4/53.2 |
| + rollout | trainval | 4096 | 72.1 | 86.3 | 9.3 | 2.9 | 2.1 | 20.7/43.3 |
| + rollout | trainval | 8192 | **72.9** | **86.5** | 18.2 | **2.8** | **2.0** | 15.3/33.8 |

- Ganancia sobre LeWM en el test completo: $63.5-39.8=23.7$ EPDMS (navtrain, sin rollout) (PDF p. 2; Tab. 2).
- El rollout y el split trainval mejoran sobre todo la precisión geodésica y el hit rate; el EPDMS en 100 escenas no sube, lo que los autores atribuyen a la varianza de 100 escenas y a que la distancia latente se alinea más con la geodésica que con las métricas NAVSIM (PDF p. 9).
- Más candidatas ⇒ mejor EPDMS y FDE, pero **menor hit rate**: con un vocabulario más denso hay más trayectorias casi idénticas a la real que compiten por el top-1 (PDF p. 9).
- Desglose por submétrica (Tab. 5, trainval + rollout, 256): NC 95.2, DAC 82.0, DDC 94.4, TLC 99.6, EP 85.2, TTC 93.3, LK 89.3, HC 92.3, EC 35.5. El EC es muy bajo en todas las variantes (13.7–43.8).

**Transferencia a imitation learning** (Tab. 3; App. A.5): ViT simple tipo Drive-JEPA, DINOv3 + proyector, 4 frames frontales $3\times256\times512$, query de trayectoria con cross-attention + MLP, MSE, fine-tuning completo.

| Configuración | V | Tipo | Frames | EPDMS |
|---|---|---|---|---|
| DINOv3 + proyector aleatorio | SV | PF | 4 | 80.2 |
| DINOv3 + proyector AD-E2E-JEPA | SV | PF | 4 | **85.4** |
| *Referencias:* Latent-WAM / WA-JEPA | MV | PF | 4 | 89.3 / 91.7 |
| *Referencias:* Transfuser / Drive-JEPA (EPDMS\*, antes del bug fix de NAVSIM) | MV / SV | PF / PB | 1 / 2 | 76.7 / 87.8 |

Las mayores subidas están en DAC (89.8 → 93.7) y EC (84.1 → 88.7) (Tab. 3).

---

## Limitaciones

**Declaradas por los autores:**
- Planificación con **objetivo oráculo**: el goal es el frame real $I_{t+F}$. Se usa para aislar la calidad del world model del aprendizaje de la política (abstract; §3.3).
- La planificación zero-shot no optimiza la seguridad; por eso se reporta EPDMS† (§3.3.2).
- 100 escenas dan resultados ruidosos (PDF p. 9).
- Solo cámara frontal "for simplicity" (§3.1).

**Observadas en la ingesta:**
- **El objetivo oráculo filtra el futuro.** Elegir la trayectoria cuyo rollout termina más cerca del embedding de $I_{t+F}$ se parece a un problema de odometría visual o de dinámica inversa: dados el presente y el futuro real, ¿qué trayectoria los une? El EPDMS zero-shot (67.3–72.9) **no es comparable** con el de las políticas de imitation learning de la Tab. 3, que no ven el futuro. Esto mide la calidad del world model, no la capacidad de conducir.
- **Sin ablación del anti-colapso.** No hay variante con proyector y $\lambda=0$ (solo stop-gradient), ni métricas de colapso (rango efectivo, varianza por dimensión). La afirmación de que SIGReg "preserva el rendimiento de planificación" (§1) no se aísla experimentalmente.
- **Sin ablación del proyector.** No se compara stride $2\times2$ frente a $1\times1$, ni distintas $D_{\text{proj}}$, ni el número de tokens.
- **Baseline LeWM poco representativo:** ViT-L desde cero (el LeWM original usa ViT-Tiny, arbitraje #12) con **batch 8** (Tab. 1). SIGReg estima la distribución sobre el batch, y con $B=8$ el test de Epps–Pulley es muy ruidoso. El mal resultado de LeWM (FDE 12–15 m) puede deberse en parte a esta configuración.
- **Confusión de hiperparámetros:** AD-E2E-JEPA usa batch 128 en navtrain y los baselines 64 o 8, todos con lr $10^{-4}$ (Tab. 1). La regla de escalado $\sqrt{\text{batch}}$ (§4.2) solo se aplica a las variantes trainval.
- $\lambda$ se ajusta "heurísticamente" con las curvas de pérdida tempranas para batches grandes (§4.2): 0.025 con batch 512, pero 0.09 con batch 256.
- **Una ejecución por configuración**, sin intervalos de confianza. En 100 escenas, un punto de hit rate es una escena.
- **IL:** solo se compara contra un proyector aleatorio. Falta el baseline DINOv3 sin proyector (parches densos) y uno con proyector preentrenado solo con SIGReg o solo con MSE.
- **SIGReg marginal por parche** (Eq. 10): con $H'W'=32$ posiciones y $N=5$ frames hay 160 tests independientes por paso. No impone independencia entre parches.

---

## Referencias cruzadas

- **Arquitectura:** [[wiki/architecture/2026_AD-E2E-JEPA_Arch|2026_AD-E2E-JEPA_Arch]]
- **Entidad:** [[AD-E2E-JEPA]]
- **Matemáticas:** [[SIGReg]] (variante parche a parche), [[Teacher_Forcing_Rollout_Loss]]
- **Baselines:** [[DINO-WM]], [[JEPA-WM]], [[LeWorldModel]] ([[wiki/papers/2026_LeWorldModel|2026_LeWorldModel]])
- **Relacionados en la bóveda:**
  - [[wiki/papers/2026_Temporal_Straightening|Temporal Straightening]]: origen del proyector convolucional sobre features congeladas.
  - [[wiki/papers/2025_LeJEPA|LeJEPA]]: SIGReg.
  - [[wiki/papers/2026_Semigroup-JEPA|Semigroup-JEPA]], [[Semigroup_Rollout_Consistency]]: otra forma de entrenar con rollout (sin TBPTT, con descuento).
  - [[wiki/papers/2026_LpWM|LpWM]], [[wiki/papers/2026_AdaJEPA|AdaJEPA]], [[wiki/papers/2025_PLDM|PLDM]]: citados como trabajo relacionado en planificación latente.
- **Antecedentes citados, no ingeridos:** JEPA-WM (Terver et al., 2025, arXiv:2512.24497), DINOv3, NAVSIM / NAVSIMv2, Drive-JEPA, WA-JEPA, AD-L-JEPA, VADv2.
- **Síntesis:** [[JEPA-master-note]]

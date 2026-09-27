---
title: "Temporal Straightening for Latent Planning"
authors: [Ying Wang, Oumayma Bounou, Gaoyue Zhou, Randall Balestriero, Tim G. J. Rudner, Yann LeCun, Mengye Ren]
year: 2026
venue: "ICML 2026 (PMLR 306)"
arxiv: "2603.12231"
source_pdf: "raw/Temporal Straightening for Latent Planning.pdf"
repo: "https://agenticlearning.ai/temporal-straightening/"
type: paper
family: jepa
modality: [video, control]
anti_collapse: [sg]
predictor: true
planner: [gd, cem]
tags: [paper, jepa, world-models, temporal-straightening, curvature, gradient-based-planning, dino-wm, planning-geometry]
---

# Temporal Straightening for Latent Planning

> [!abstract] TL;DR
> Un world model JEPA (encoder + encoder de acciones + predictor ViT) entrenado con $\mathcal L_{\text{pred}}$ + un **regularizador de curvatura** $\mathcal L_{\text{curv}} = 1-\cos(v_t, v_{t+1})$ sobre velocidades latentes consecutivas produce trayectorias latentes más rectas. En ese espacio la distancia euclídea se aproxima mejor a la geodésica y el objetivo de planificación está **mejor condicionado** (Thm 4.4, dinámica lineal), de modo que un planificador por **gradient descent** supera claramente a DINO-WM con el mismo planificador: +20–60 puntos en open-loop y +20–30 en MPC (PDF §1; Tab. 1).

> [!warning] Procedencia
> - **Autoría:** Ying Wang, Oumayma Bounou, Gaoyue Zhou, Yann LeCun\*, Mengye Ren\* (NYU); Randall Balestriero (Brown); Tim G. J. Rudner (Toronto). \*Equal advising (PDF p. 1).
> - **Venue:** *Proceedings of the 43rd ICML*, Seúl, PMLR 306, 2026 (PDF p. 1). arXiv:2603.12231v3 (11 Aug 2026).
> - **Tipo:** empírico con un resultado teórico en el caso lineal.
> - **Sin señales de alerta:** autoría coherente con la línea del grupo (Zhou es primer autor de DINO-WM; Ren dirige el lab de [[AdaJEPA]]), experimentos con 3 semillas, venue revisado. Código anunciado en la página del proyecto; no se indica repositorio GitHub.

---

## Resumen ejecutivo

**Problema.** Las features preentrenadas (DINOv2) son semánticamente ricas pero no están pensadas para planificar: sus trayectorias latentes son muy curvas, la distancia euclídea al objetivo no refleja el progreso geodésico y el paisaje de la pérdida de planificación es muy no convexo, por lo que los planificadores por gradiente se atascan y la práctica recurre a CEM/MPPI, caros en latencia (PDF §1; Fig. 1, 2).

**Propuesta.** Inspirados en la *perceptual straightening hypothesis* (Hénaff et al., 2019), entrenan conjuntamente encoder y predictor con una penalización de la curvatura local de las trayectorias latentes (PDF §3.2–3.3).

**Hallazgos** (PDF §5.2):
1. La pérdida JEPA por sí sola ya induce un **straightening implícito**.
2. El regularizador explícito lo refuerza y lo estabiliza.
3. Tras el straightening, la distancia euclídea latente se alinea con la geodésica (A\*), aunque el entrenamiento use trayectorias subóptimas (Fig. 6).
4. Features muy comprimidas ($14\times14\times8$) bastan para reconstruir las observaciones con alta fidelidad.

---

## Arquitectura

Detalle completo en [[wiki/architecture/2026_Temporal_Straightening_Arch|arquitectura Temporal Straightening]].

- **Sensory encoder** $\mathcal E^s_\phi$: dos variantes (PDF §5.1): (i) DINOv2 **congelado** + proyector CNN ligero **entrenable** $\mathcal P_\phi$, $z^v_t=\mathcal P_\phi(e_t)\in\mathbb R^{m_v\times d_v}$ (Eq. 13); (ii) **ResNet desde cero**.
- **Action encoder** $\mathcal E^a_\psi:\mathbb R^{n_a}\to\mathbb R^{d_a}$; propiocepción $\mathcal E^p_\xi$ opcional.
- **Predictor** $f_\theta$: ViT con máscara causal temporal, historia de $K$ frames (Eq. 2); 3 frames de historia, frameskip 5 (Tab. 3).
- **Cabeza de agregación** $h_\phi$ (MLP, salida 128) sobre la que se aplica $\mathcal L_{\text{curv}}$ en los experimentos principales; $\mathcal L_{\text{pred}}$ se aplica a las features espaciales (App. B.6, Fig. 13).
- **Anti-colapso:** solo **stop-gradient en la rama target, sin EMA** (PDF §3.3). Los autores reconocen que el colapso sigue siendo posible en teoría.

---

## Formulación

Notación del paper: $z_t\in\mathbb R^d$ latente, $K$ = historia del predictor **y también** horizonte de planificación en §4 (doble uso), $T$ / $H$ = horizonte en §5. Equivalencia con la notación canónica: $d\equiv D$; horizonte de planificación $\equiv H$.

- Velocidades latentes $v_t=z_{t+1}-z_t$ (Eq. 3); curvatura vía $\mathcal C=\frac{v_t\cdot v_{t+1}}{\|v_t\|_2\|v_{t+1}\|_2}$ (Eq. 4).
- $\mathcal L_{\text{pred}}=\|\hat z_{t+1}-\mathrm{sg}(z_{t+1})\|_2^2$ (Eq. 5), $\mathcal L_{\text{curv}}=1-\mathcal C$ (Eq. 6).
- $\mathcal L_{\text{total}}=\mathcal L_{\text{pred}}+\lambda\,\mathcal L_{\text{curv}}$, $\lambda\ge0$ — **forma aditiva** (Eq. 7). $\lambda=0.1$ en todas las features espaciales; para las globales, $\lambda\in\{0.1,0.01,0.001\}$ elegido por MPC en 2 semillas de validación (Tab. 1).
- Página matemática: [[Temporal_Straightening_Loss]].
- Teoría (dinámica lineal $z_{t+1}=Az_t+Ba_t$): el número de condición efectivo del Hessiano de planificación está acotado por $\kappa(B)^2\big(\frac{1+\varepsilon}{1-\varepsilon}\big)^{2(K-1)}$ si $\|A-I\|_2=\varepsilon<1$ (Thm 4.4, Eq. 12); y un coseno medio alto implica $(A-I)$ pequeño **en las direcciones visitadas** (Prop. C.9). Página: [[Planning_Hessian_Conditioning]].

---

## Resultados clave

Protocolo: 50 episodios de test, media ± std sobre 3 semillas de datos; objetivo alcanzable en 25 pasos, frameskip 5 ⇒ $H=5$ llamadas al predictor; GD con Adam, lr 0.1, 100 pasos, inicialización a cero (PDF §5.3; Tab. 4). Entornos: Wall, PointMaze UMaze y Medium, PushT (PDF §5).

**Tab. 1 (GD), features espaciales, éxito (%)** — mejor configuración con $\mathcal L_{\text{curv}}$ frente a DINO-WM (DINOv2 patch, $14\times14\times384$):

| Entorno | DINO-WM OL / MPC | Mejor con straightening OL / MPC | Fuente |
|---|---|---|---|
| Wall | 52.67 / 76.67 | 90.67 (proj) / 100.00 (proj y ResNet) | Tab. 1 |
| PointMaze-UMaze | 35.33 / 80.67 | 94.00 (proj) / 100.00 (proj) | Tab. 1 |
| PointMaze-Medium | 40.83 / 76.67 | 82.67 (proj) / 99.33 (ResNet) | Tab. 1 |
| PushT | 56.00 / 66.00 | 77.33 (proj) / 91.33 (ResNet) | Tab. 1 |

- **Implícito vs. explícito:** entrenar el proyector sin $\mathcal L_{\text{curv}}$ ya mejora a DINO-WM (p. ej. Wall OL 80.00); el término explícito añade >10 puntos en muchos setups (Tab. 1; PDF p. 8).
- **Curvatura ↔ éxito:** a igual tipo de encoder, menor curvatura (mayor coseno) ⇒ mayor éxito en GD open-loop (Fig. 5).
- **GD vs. CEM** (open-loop, CEM con 200 muestras y 10 iteraciones): el straightening mejora ambos y reduce la brecha; CEM es ~10× más lento en wall-clock (App. B.3; Tab. 5, Fig. 10). Ej.: UMaze proj+curv GD 94.00 = CEM 94.00; PushT ResNet+curv GD 70.67 vs CEM 72.67 (Tab. 5).
- **Dimensión:** $d_v\in\{8,32\}$ óptimo; $d_v=2$ insuficiente, $d_v=128$ empeora (App. B.4, Fig. 11). Colapsar a un vector global empeora la predicción; ResNet desde cero da mejores features globales que un proyector global sobre DINO (PDF p. 8).
- **Variantes del coseno espacial** (patch / mean / flatten / agg): todas mejoran a "none"; *agg* es la mejor en Wall, UMaze y PushT (App. B.6, Fig. 14).
- **Otras regularizaciones temporales** ($\mathcal L_{\text{smooth}}=\mathbb E_t\|z_{t+1}-z_t\|_2^2$, contrastivo temporal InfoNCE con ventana $k\in\{2,5\}$): no mejoran PushT; el straightening sí (App. B.5, Fig. 12).
- **Horizonte largo (50 pasos, Tab. 2):** PointMaze-Medium MPC 65.33 (DINO-WM) → 98.67 (ResNet+curv); PushT MPC 27.33 → 33.33. Añadir un coste global $\mathcal L_{\text{plan}}=\mathcal L_{\text{spatial}}+0.1\,\mathcal L_{\text{agg}}$ mejora el MPC (proj PushT 24.00 → 33.33; ResNet PushT 33.33 → 36.00; proj Medium 88.00 → 92.00) (Tab. 2). El texto dice "across all models", pero ResNet en Medium queda igual (98.67 → 98.67).
- **Teleported-PointMaze:** con straightening el agente aprovecha la teletransportación (estados lejanos en píxeles, cercanos en tiempo); sin él se atasca (App. F, Fig. 27). Resultado cualitativo (un caso representativo).

---

## Limitaciones

**Declaradas por los autores** (PDF p. 9, §4):
- Coste de objetivo euclídeo **simétrico**: subóptimo con dinámica asimétrica o irreversible; sugieren cuasimétricas.
- Garantía teórica solo para **dinámica lineal**; el caso no lineal (productos de Jacobianos dependientes del estado) queda como trabajo futuro.
- Rollouts largos acumulan error (fallos de PushT a 50 pasos, Fig. 9).
- Colapso posible en teoría con stop-grad.

**Observadas en la ingesta:**
- El Thm 4.4 exige $d_a=d$ y $B$ invertible; en **todos** los experimentos $d_a<d$ (acción 2-D), así que se aplica solo la Remark 4.5, que requiere hipótesis de controlabilidad adicionales no verificadas. El puente coseno → $\|A-I\|$ (Prop. C.9) supone $\|v_t\|_2$ **constante** y cubre solo direcciones visitadas (Remark C.10).
- La cota $e^{6\varepsilon K}$ es cuantitativamente débil: con $\varepsilon=0.5$, $K=5$ da $\kappa(B)^2e^{15}$. Su valor es cualitativo (crecimiento lento con el horizonte cuando $\varepsilon\to0$).
- **Confusión de hiperparámetros:** sin straightening se usa lr del encoder $10^{-6}$ en lugar de $10^{-5}$ por "severe performance degradation" (Tab. 3, nota a). Los baselines $\lambda=0$ y los modelos con curvatura no comparten lr.
- Sin curvatura, ResNet espacial se degrada anómalamente (Wall OL 1.33, Tab. 1). Los autores lo atribuyen a la altísima curvatura; no se reporta ninguna métrica de colapso.
- $\lambda$ de las features globales se escoge por entorno con MPC en validación (Tab. 1), lo que favorece ligeramente a la variante regularizada.
- *agg* no es la mejor en PointMaze-Medium: allí gana *flatten* (Fig. 14c).
- DINO-WM se evalúa aquí con **GD**, no con su CEM original; con CEM (Tab. 5) la ventaja se reduce mucho (p. ej. PushT: 71.33 DINO-WM vs 72.67 ResNet+curv).
- Solo entornos 2-D de simulación, visualmente simples.

---

## Referencias cruzadas

- **Arquitectura:** [[wiki/architecture/2026_Temporal_Straightening_Arch|2026_Temporal_Straightening_Arch]]
- **Entidad:** [[Temporal_Straightening]]
- **Matemáticas:** [[Temporal_Straightening_Loss]], [[Planning_Hessian_Conditioning]]
- **Baseline:** [[DINO-WM]]
- **Relacionados en la bóveda:**
  - [[wiki/papers/2026_Semantic_Tube|Semantic Tube]]: la misma pérdida $1-\cos$ de diferencias consecutivas, aplicada a estados ocultos de LLMs.
  - [[wiki/papers/2026_MotionJEPA|MotionJEPA]]: mide la rectitud de trayectorias con $S(z)$.
  - [[wiki/papers/2026_AdaJEPA|AdaJEPA]]: mismo lab y mismo planificador GD (Adam, lr 0.1, 100 pasos).
  - [[wiki/papers/2025_PLDM|PLDM]]: origen del entorno Wall.
  - [[wiki/papers/2026_LeWorldModel|LeWorldModel]], [[wiki/papers/2026_Semigroup-JEPA|Semigroup-JEPA]]: encoder entrenado junto al predictor frente a encoder congelado.
- **Síntesis:** [[JEPA-master-note]]

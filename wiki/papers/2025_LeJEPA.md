---
title: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
authors: [Randall Balestriero, Yann LeCun]
year: 2025
tags: [jepa, self-supervised, theory, sigreg, collapse-free, foundation-models]
---

# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

**Resumen**:
Este trabajo seminal de Randall Balestriero y Yann LeCun sienta las bases teóricas y prácticas de los modelos Joint-Embedding Predictive Architectures (JEPA) libres de heurísticas. Introduce **LeJEPA** (*Latent-Euclidean JEPA*), un marco de preentrenamiento autosupervisado que demuestra rigurosamente que la **distribución Gaussiana isotrópica** $\mathcal{N}(0, I_K)$ es la distribución óptima que deben seguir los embeddings para minimizar el riesgo de error en tareas *downstream*. Para imponer esta distribución sin incurrir en la maldición de la dimensionalidad ni en inestabilidades numéricas, proponen **SIGReg** (*Sketched Isotropic Gaussian Regularization*), un regularizador basado en proyecciones unidimensionales aleatorias (Teorema de Cramér-Wold) y la prueba de bondad de ajuste de **Epps-Pulley**.

**Contribuciones Principales**:
1. **Prueba de Optimalidad de la Distribución Gaussiana Isotrópica**: Demuestran formalmente (para pruebas lineales, k-NN radial y métodos de kernel como Nadaraya-Watson) que la distribución Gaussiana isotrópica es el único minimizador del sesgo cuadrático integrado (*Integrated Square Bias*, ISB) bajo restricciones de covarianza escalar (ver [[Isotropic_Gaussian_Optimality]]).
2. **SIGReg (Sketched Isotropic Gaussian Regularization)**: Formulación de la regularización anti-colapso como un contraste de hipótesis estadísticas ($H_0: P_\theta = Q$). Al proyectar sobre $M$ direcciones aleatorias $a_m \in \mathbb{S}^{K-1}$ y evaluar la función característica empírica con la estadística de Epps-Pulley, se obtienen gradientes uniformemente acotados ($\mathcal{O}(1/N)$) y complejidad lineal $\mathcal{O}(N)$ en tiempo y memoria (ver [[SIGReg]]).
3. **Eliminación Total de Heurísticas**: Se descartan componentes ad-hoc comúnmente utilizados para estabilizar el entrenamiento: *stop-gradients*, redes *teacher-student* con esquemas EMA, proyectores/predictores asimétricos, blanqueamiento explícito o capas de *register tokens*. La arquitectura se reduce a un encoder y un proyector simple, con un único hiperparámetro de balance ($\lambda$).
4. **Pérdida de Entrenamiento Predictiva del Rendimiento Downstream**: A diferencia de otros métodos SSL donde la pérdida de entrenamiento no correlaciona con la calidad de representación, la pérdida combinada de LeJEPA exhibe una correlación de Spearman $\rho_s \approx 94\% - 99\%$ con la precisión de evaluación lineal congelada (*frozen linear probing*), permitiendo selección de modelos y validación cruzada sin etiquetas (*label-free model selection*).
5. **Escalabilidad y Preentrenamiento In-Domain**: Validado en más de 60 arquitecturas (ViT, ResNet, ConvNeXt, Swin, MaxViT) y escalas de hasta 1.8 mil millones de parámetros (ViT-gigantic). Muestra que el preentrenamiento *in-domain* con LeJEPA en conjuntos de datos específicos y pequeños (e.g. Galaxy10, Food101) supera a modelos fundacionales gigantes como DINOv2/v3 transferidos.

**Detalles de Arquitectura**:
Ver [[2025_LeJEPA]] en la sección de arquitecturas.

**Fundamentos Matemáticos**:
- Pérdida general: $\mathcal{L}_{\text{LeJEPA}} = \lambda \frac{1}{V}\sum_{v=1}^V \text{SIGReg}(\{z_{n,v}\}_{n=1}^B) + \frac{1-\lambda}{B}\sum_{n=1}^B \mathcal{L}_{\text{pred}}^{(V_g)}(\{z_{n,v}\}_{v=1}^V)$ (ver [[LeJEPA_Loss]]).
- Regularizador: [[SIGReg]].
- Demostración de optimalidad de la Gaussiana: [[Isotropic_Gaussian_Optimality]].
- Pérdida de invariancia multi-vista: [[Invariance_Loss]].

**Conexión con Desarrollos Posteriores**:
- [[2026_LeWorldModel]]: Extensión de LeJEPA a Modelos de Mundo *end-to-end* con dinámicas latentes condicionadas por acciones.
- [[2026_LeVJEPA]]: Adaptación de LeJEPA al preentrenamiento de video con atención *block-causal* y *token dropping* extremo (95%).
- [[2026_LeJEPA_Identifiability]]: Marco teórico complementario que analiza la identificabilidad lineal y unicidad de las representaciones aprendidas por LeJEPA.

---
title: LeJEPA
tags: [architecture, entity, jepa, foundation-models, sigreg]
---

# LeJEPA (Latent-Euclidean JEPA)

**LeJEPA** es la arquitectura y marco fundacional de aprendizaje autosupervisado presentado por Randall Balestriero y Yann LeCun en 2025 ([[2025_LeJEPA]]). Establece la primera formulación teóricamente demostrada y libre de heurísticas para modelos Joint-Embedding Predictive Architectures (JEPA).

---

## 1. Axiomas y Principios de Diseño

LeJEPA formula el aprendizaje de representaciones visuales a partir de dos condiciones mínimas necesarias:
1. **Predicción / Invariancia de Vistas**: Mapear representaciones de vistas locales/aumentadas $z_{n,v}$ hacia el centroide de vistas globales $\mu_n$ mediante un error cuadrático medio simple ([[Invariance_Loss]], [[LeJEPA_Loss]]).
2. **Restricción de Distribución Óptima**: Forzar que el espacio de embedding siga una **distribución Gaussiana isotrópica** $\mathcal{N}(0, I_K)$, demostrada matemáticamente como la configuración que minimiza el sesgo y la varianza en tareas downstream ([[Isotropic_Gaussian_Optimality]]).

---

## 2. Innovación: Eliminación de Heurísticas

Históricamente, los modelos SSL y JEPA (BYOL, DINO, I-JEPA, V-JEPA) requerían una delicada combinación de heurísticas asimétricas para evitar el colapso de la representación:
- Redes *Teacher-Student* acopladas con actualización por promedio móvil exponencial (EMA).
- Operaciones de corte de gradiente (*stop-gradient*).
- Redes *Predictor* de capacidad limitada o enmascaramiento estructurado complejo.
- Capas de *Register Tokens* o blanqueamiento explícito.

LeJEPA elimina por completo todos estos componentes mediante **[[SIGReg]]**, permitiendo:
- Entrenar un único encoder $f_\theta$ con gradientes simétricos.
- Reducir el pipeline a un único hiperparámetro de balance $\lambda \in [0, 1]$ (valor robusto por defecto $\lambda = 0.05$).
- Complejidad lineal $\mathcal{O}(N)$ en tiempo y memoria.

---

## 3. Correlación de la Pérdida de Entrenamiento

Una de las propiedades más distintivas de LeJEPA es que su pérdida de entrenamiento $\mathcal{L}_{\text{LeJEPA}}$ presenta una correlación de Spearman de hasta el **99%** con la precisión de evaluación lineal congelada (*frozen linear probe accuracy*). Esto resuelve el problema histórico en SSL donde la pérdida de entrenamiento no permitía comparar puntos de control ni realizar selección de modelos sin etiquetas.

---

## 4. Evolución y Derivaciones en la Literatura

El marco LeJEPA ha originado una familia completa de arquitecturas en la literatura moderna de World Models:
- [[2025_LeJEPA]]: El paper fundacional en imágenes estáticas y visión general.
- [[2026_LeWorldModel]]: Integración de LeJEPA con dinámicas latentes de transición y condicionamiento por acciones para modelos de mundo *end-to-end*.
- [[2026_LeVJEPA]]: Extensión a secuencias de video con *block-causal attention* y *uniform random token dropping* del 95%.
- [[2026_LeJEPA_Identifiability]]: Pruebas matemáticas de identificabilidad lineal y unicidad del espacio latente.

---

## 5. Referencias en la Wiki
- Resumen del paper: [[2025_LeJEPA]]
- Diagrama y flujo de arquitectura: [[2025_LeJEPA]] (sección Architecture)
- Fundamentación matemática: [[SIGReg]], [[LeJEPA_Loss]], [[Isotropic_Gaussian_Optimality]], [[Invariance_Loss]]

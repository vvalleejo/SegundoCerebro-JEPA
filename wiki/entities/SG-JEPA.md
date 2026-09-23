---
title: "SG-JEPA (Semigroup Joint-Embedding Predictive Architecture)"
tags: [entity, architecture, jepa, world-models, physics-generalization, semigroup, sigreg]
---

# SG-JEPA (Semigroup Joint-Embedding Predictive Architecture)

## Descripción
**SG-JEPA (Semigroup-JEPA)** (2026, Yale / Jump Trading / Brown University; autores: Andy Zeyi Liu, Haoran Sun, Lucas Baker, Randall Balestriero, John Sous) es una arquitectura de modelo de mundo basada en incrustaciones conjuntas que extiende el marco [[LeWorldModel]] para lograr **generalización física fuera de distribución (Zero-Shot Physics Generalization)** y mitigar la deriva de error en predicciones a horizontes largos.

A diferencia de los modelos previos que predicen a 1 solo paso bajo un conjunto fijo de reglas dinámicas, SG-JEPA suministra parámetros físicos rectores (como el campo gravitatorio $g$) mediante el canal de acción y entrena conjuntamente el encoder visual y el predictor latente a través de un **rollout autorregresivo recursivo de $K$ pasos**, modelado como un semigrupo discreto de transformaciones y regularizado con [[SIGReg]].

---

## Componentes Arquitectónicos
1. **Encoder Visual Compartido sin Stop-Gradient**: Un Vision Transformer (ViT-Tiny) que extrae un vector latente $z_t \in \mathbb{R}^{256}$ proyectando el token `[CLS]`. El mismo encoder genera los estados de contexto y los objetivos futuros $z_{t+k}$ sin ramas EMA congeladas, permitiendo que el gradiente del rollout moldee directamente el subespacio latente mientras SIGReg previene el colapso.
2. **Canal de Fusión Física ($\tilde{a}_t = [u_t; g]$)**: Inyecta el escalar de gravedad $g$ junto con los controles de actuación, procesándolos mediante una convolución temporal 1D y un MLP con activaciones SiLU para emitir embeddings de contexto $c_t \in \mathbb{R}^{256}$.
3. **Predictor de Historia Latente**: Procesa una ventana de historia de $H=20$ pasos para predecir el siguiente estado latente. Admite arquitecturas de **Residual GRU** (3 capas residuales de ancho 512), **SSM (Mamba S6)** o **Causal Transformer con AdaLN**.
4. **Objetivo de Semigrupo con Rollout Descontado**: Minimiza la pérdida autorregresiva $\mathcal{L}_{\text{roll}} = \sum_{k=1}^K w_k \|\hat{z}_{t+k} - z_{t+k}\|_2^2$ ponderada con decaimiento exponencial ($w_k \propto \gamma^{k-1}$, $\gamma = 0.95$), combinada aditivamente con [[SIGReg]].
5. **Políticas de Difusión en Bucle Cerrado**: Permite entrenar Diffusion Policies sobre las representaciones latentes congeladas comprimidas por una GRU causal, alcanzando mejoras de hasta $2.5\times$ en tareas robóticas de contacto y captura balística.

---

## Relevancia para el Doctorado
SG-JEPA es un pilar conceptual fundamental para la tesis en World Models basados en JEPA:
- **Clausura Predictiva**: Introduce la formulación teórica de la matriz de clausura $C_W(g) = W T(g) - A_W(g)W$, demostrando que el rollout multi-paso fuerza al encoder a retener subespacios dinámicamente invariantes.
- **Inversión de Ranking**: Prueba matemáticamente por qué los métodos entrenados con teacher-forcing a 1 paso (como LeWM estándar) seleccionan representaciones subóptimas para trayectorias de horizonte extendido.
- **Extrapolación Física**: Establece cotas rigurosas de error OOD en función de la matriz de momentos de la ley física $M_\psi$ y el factor de cobertura $L_{\text{law}}(g^\star)$.

---

## Enlaces Relacionados
- **Paper**: [[2026_Semigroup-JEPA]]
- **Arquitectura Detallada**: [[2026_Semigroup-JEPA]]
- **Fundamentos Matemáticos**: [[Semigroup_Rollout_Consistency]], [[Latent_Dynamics_Consistency]]
- **Pérdidas y Regularizadores**: [[SIGReg]], [[LeWM_Loss]]
- **Arquitecturas Conexas**: [[LeWorldModel]], [[DINO-WM]], [[V-JEPA2]], [[AdaJEPA]], [[SkyJEPA]]

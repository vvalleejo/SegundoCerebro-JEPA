---
title: "MotionJEPA (Visual Change-Aware Joint-Embedding Predictive Architecture)"
tags: [entity, architecture, motion-jepa, disreg, jepa, world-models, anti-collapse]
---

# MotionJEPA (Visual Change-Aware Joint-Embedding Predictive Architecture)

## Descripción
**MotionJEPA** (2026, Universidad de Oxford, vivo Tech Research, Brown University y AMI Labs; autores: Markus Karmann, Shile Li, Christian Internò, Bruno Andreis, David Klindt, Randall Balestriero, Jindong Gu, Philip Torr et al.) es una arquitectura de modelo de mundo basada en incrustaciones conjuntas que resuelve el **sesgo hacia características lentas (*slow-feature bias*)** y el **colapso temporal de características** en el modelado visual sin reconstrucción de píxeles.

Introduce el regularizador **DISReg** (*Difference Image and Single image embedding Regularization*), el cual acopla un encoder de imágenes de diferencia temporal con un predictor inverso de diferencias latentes, garantizando que el modelo capture simultáneamente información estática global del entorno y dinámicas rápidas de cambio visual.

---

## Componentes Arquitectónicos Clave
1. **Encoder Visual ViT-Tiny ($e_\theta$)**: Proyecta la observación $o_t$ a un embedding de estado $z_t \in \mathbb{R}^{192}$ mediante el token `[CLS]` y un proyector MLP con BatchNorm.
2. **Predictor Forward Causal ($p_\phi$)**: Modelo autorregresivo basado en Transformer de 6 capas que predice $\hat{z}_{t+1}$ condicionado por acciones.
3. **Módulo DISReg**:
   - **Encoder de Diferencias ($	ext{DiffEnc}_\alpha$)**: Codifica $o_{t+1} - o_t \to d_t$, donde los fondos estáticos desaparecen numéricamente.
   - **Predictor de Cambio Latente ($	ext{DiffPred}_\beta$)**: Predice $\hat{d}_t$ a partir de $[z_t; z_{t+1}]$ sin reconstruir píxeles ni requerir etiquetas de acción.
   - **Regularización SIGReg Dual**: Aplica [[SIGReg]] tanto al estado $z$ (término estático) como a la diferencia $d$ (término dinámico).
4. **Descarte de Módulos Auxiliares en Inferencia**: Los componentes de diferencia son exclusivos del bucle de entrenamiento, dejando el modelo de despliegue con cero sobrecarga computacional.

---

## Relevancia para el Doctorado
- **Solución al Colapso ante Distractores Estáticos**: Resuelve el problema identificado por Sobal et al. (2022) donde las arquitecturas JEPA se obsesionan con texturas o ruidos constantes descartando dinámicas críticas.
- **Independencia de Acciones**: A diferencia de SMWM y Delta-JEPA, funciona en dinámicas pasivas no actuadas, siendo ideal para preentrenamiento autosupervisado generalista.
- **Trayectorias Latentes Rectificadas**: Demuestra que predecir diferencias temporales endereza las geodésicas en el espacio latente, facilitando la planificación por optimización basada en gradiente o muestreo CEM.

---

## Enlaces Relacionados
- **Paper**: [[2026_MotionJEPA]]
- **Arquitectura Detallada**: [[2026_MotionJEPA]]
- **Concepto Matemático**: [[DISReg]], [[SIGReg]], [[LeWM_Loss]]
- **Modelos Conexos**: [[LeWorldModel]], [[2026_Semigroup-JEPA]], [[V-JEPA2]]
- **Repositorio**: [https://github.com/mkarmann/motion-jepa](https://github.com/mkarmann/motion-jepa)

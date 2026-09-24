---
title: "VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models"
authors: [Yongchao Huang]
year: 2026
venue: "arXiv preprint"
arxiv: "2601.14354"
source_pdf: "raw/VJEPA Variational Joint Embedding Predictive Architectures as Probabilistic World Models.pdf"
repo: "https://github.com/YongchaoHuang/VJEPA"
type: "paper"
family: "jepa"
modality: [theory]
anti_collapse: [ema-sg, kl]
predictor: true
planner: [none]
tags: [paper, jepa, variational, probabilistic, world-models, bjepa]
---

# VJEPA y BJEPA: Probabilistic World Models

## Resumen Ejecutivo
Este paper (2026) presenta **Variational JEPA (VJEPA)**, una generalización probabilística de la arquitectura JEPA estándar que permite aprender una distribución predictiva sobre futuros estados latentes usando un objetivo variacional. También introduce **Bayesian JEPA (BJEPA)**, una extensión que factoriza la creencia predictiva usando un *Product of Experts (PoE)*, separando la dinámica del entorno de los priors (objetivos/restricciones).

## Conceptos Clave
- **De Determinista a Probabilístico**: Las JEPAs tradicionales (como [[LeWorldModel]]) predicen un punto estimado en el espacio latente. VJEPA predice una *distribución* (ej. Gaussiana con media y varianza), lo que permite estimar la incertidumbre y manejar futuros multimodales.
- **Invarianza a "Nuisance" (Noisy TV)**: Al no tener que reconstruir los píxeles de observación (al contrario que los autoencoders o modelos generativos tradicionales), VJEPA filtra las variables distractoras de alta entropía (ruido) y captura únicamente la información mutua predecible.
- **BJEPA y Product of Experts (PoE)**: BJEPA separa el modelo en un experto de *Likelihood* (dinámicas aprendidas) y un experto *Prior* (restricciones o metas). La inferencia se realiza fusionando ambos mediante PoE, lo que equivale matemáticamente a un filtro Bayesiano latente.

## Formulación Matemática (Ver [[VJEPA_Loss]])
El objetivo de entrenamiento minimiza el Negative Log-Likelihood (NLL) más un término de regularización KL:
$$ \mathcal{L}_{VJEPA} = \mathbb{E} \left[ - \log p_\phi(Z_T \mid Z_C, \xi_T) \right] + \beta \mathbb{E} \left[ \text{KL}(q_{\theta'}(Z_T \mid x_T) \parallel p(Z_T)) \right] $$

## Impacto para World Models
- Proporciona garantías formales para evitar el colapso de representaciones sin necesidad de heurísticas arquitectónicas (el colapso se evita por la asimetría de la información y la divergencia KL).
- Habilita la propagación de creencias (belief propagation) puramente en el espacio latente, permitiendo planificación estocástica robusta.

## Referencias Cruzadas
- **Arquitectura**: [[wiki/architecture/2026_VJEPA|2026_VJEPA]]

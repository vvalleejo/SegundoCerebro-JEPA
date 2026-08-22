---
title: "HP-JEPA: Hierarchical Partitioning for Multi-Resolution Graph Joint-Embedding Predictive Learning"
authors: [Ruichen Xu, Jingxiang Qu, Wenhan Gao, Jiaxing Zhang, Linsey Pang, Ravid Shwartz-Ziv, Yann LeCun, Yuefan Deng (Stony Brook, TikTok, PayPal, NYU)]
year: 2026
tags: [paper, hp-jepa, jepa, graphs, gnn, self-supervised, multi-resolution]
---

# HP-JEPA: Partición Jerárquica Multirresolución en Grafos con JEPA

## Resumen Ejecutivo
Este paper (Agosto 2026) introduce **HP-JEPA** (Hierarchical Partitioning Joint-Embedding Predictive Architecture), extendiendo el paradigma JEPA al **aprendizaje autosupervisado sobre datos estructurados en grafos** a múltiples escalas de resolución. Aborda la principal limitación de los modelos previos de grafos tipo JEPA (como *Graph-JEPA*), que dependían de una única partición fija de granularidad, perdiendo dependencias locales finas (motivos químicos) o topologías globales.

## El Problema: El Sesgo de Resolución Fija en Grafos
Los grafos del mundo real (moléculas, redes de proteínas, redes sociales) son inherentemente **multigranulares**:
- **Particiones finas**: Preservan motivos locales y enlaces de corto alcance, pero fragmentan la organización regional y las dependencias estructurales de largo alcance.
- **Particiones gruesas**: Capturan la topología global y el contexto regional, pero difuminan los detalles locales discriminativos.

Los métodos contrastivos dependen de aumentaciones heurísticas y pares negativos, mientras que los autoencoders de grafos (GraphMAE) reconstruyen nodos o aristas crudas. *Graph-JEPA* introdujo predicción latente sobre parches, pero limitado a una sola granularidad fija.

## Metodología y Arquitectura de HP-JEPA
HP-JEPA divide el grafo en un banco ordenado de resoluciones de partición de grueso a fino ($K_\ell = 2^\ell$):

1. **Tokenización de Grafo Multirresolución**: Cada resolución $\ell \in \{1, \dots, L\}$ particiona el conjunto de nodos $V$ de forma independiente en subgrafos/parches conexos con descriptores posicionales estructurales $\rho_i^{(\ell)}$ (basados en Random Walks).
2. **Predicción de Incrustación Conjunta por Resolución**:
   - **Encoder online** $f_\theta$ procesa los parches de contexto visibles junto con sus consultas estructurales.
   - **Target Encoder EMA** $\bar{f}_\theta$ procesa los parches objetivo (target) sin ver las consultas estructurales.
   - **Predictor** $q_\psi$ predice la representación del target a partir del contexto y la posición estructural del target.
   - **Mapeo al Hiperboloide de Lorentz**: Mapea la media escalar $m(u) = d^{-1}\mathbf{1}^\top u$ de cada estado target a una coordenada bidimensional $\Phi(u) = [\cosh(m(u)), \sinh(m(u))]^\top \in \mathbb{H}_+^1$ optimizada con Smooth $L_1$.
3. **Readout Multirresolución Ponderado**:
   - Agrega los tokens de cada resolución mediante *mean pooling* obteniendo $\{h_G^{(\ell)}\}_{\ell=1}^L$.
   - Combina las resoluciones bien por concatenación directa ($h_G^{\text{concat}}$) o mediante **pesos adaptativos por tarea** ($h_G^{\text{task}} = \sum_{\ell=1}^L \omega_\ell^{\text{task}} h_G^{(\ell)}$) con suavizado uniforme $\lambda_{\text{unif}}$, permitiendo al modelo downstream elegir qué escalas son relevantes manteniendo el encoder congelado.

## Garantías Teóricas
- **Propiedad Isométrica**: Demuestran que la función $\Phi(u)$ es una codificación isométrica bajo distancia geodésica hiperbólica de Lorentz respecto a la proyección de la media de características.
- **Límite de Oráculo Adaptativo a la Resolución (Teorema 1)**: Mediante complejidad de Rademacher, demuestran que el espacio de ponderación de resoluciones de HP-JEPA compite con el mejor predictor lineal de resolución fija elegido en retrospectiva, con una penalización que escala solo logarítmicamente con el número de resoluciones $\mathcal{O}(\sqrt{\log L / N})$.

## Resultados Empíricos
- **Clasificación y Regresión de Grafos**: Supera a *Graph-JEPA* en 6 de 8 benchmarks (PROTEINS, MUTAG, DD, REDDIT-BINARY, REDDIT-MULTI-5K, IMDB-BINARY, IMDB-MULTI y ZINC-12K).
- **Desentrelazamiento Topológico**: En el dataset biológico PROTEINS, HP-JEPA logra separar de forma natural micro-proteínas no enzimáticas en un subespacio estructurado según el tamaño del grafo sin usar etiquetas supervisadas.

## Relevancia para el Doctorado
HP-JEPA amplía la familia de World Models de Yann LeCun al dominio de estructuras topológicas no euclidianas. Demuestra cómo formular jerarquías espaciales y multigranulares dentro de la predicción latente, un concepto directamente aplicable a World Models jerárquicos y relacionales.

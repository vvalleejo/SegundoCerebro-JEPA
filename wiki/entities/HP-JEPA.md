---
title: "HP-JEPA (Hierarchical Partitioning Graph JEPA)"
tags: [entity, architecture, jepa, graphs, gnn, multi-resolution]
---

# HP-JEPA (Hierarchical Partitioning Graph JEPA)

## Descripción
**HP-JEPA** (2026, Stony Brook / TikTok / NYU) es un framework de aprendizaje autosupervisado para **grafos** basado en [[JEPA]], que supera el sesgo de granularidad fija mediante particionamiento jerárquico y predicción de incrustación conjunta multirresolución.

## Componentes Arquitectónicos
1. **Multi-Resolution Graph Tokenizer**: Descompone el grafo de entrada en un banco de resoluciones de partición de granularidad creciente ($K_\ell = 2^\ell$). Para cada parche, extrae tokens de contenido visual/estructural $z_i^{(\ell)}$ junto con descriptores de posición relativa en el grafo $\rho_i^{(\ell)}$.
2. **Resolution-Wise Predictor & Lorentz Hyperboloid Target**: En cada resolución activa, un codificador de contexto online y un codificador target EMA interactúan con un predictor latente que aproxima la media proyectada en la rama positiva del hiperboloide de Lorentz ($\mathbb{H}_+^1$) vía Smooth $L_1$.
3. **Task-Specific Resolution Readout**: Genera representaciones a nivel de grafo agregando los tokens por resolución $\{h_G^{(\ell)}\}_{\ell=1}^L$ y combinándolos dinámicamente mediante pesos de resolución aprendidos por tarea ($\omega_\ell^{\text{task}}$), adaptándose a la escala óptima de cada problema sin reentrenar el encoder.

## Relevancia para el Doctorado
HP-JEPA amplía la teoría y aplicación de JEPA a **estructuras de grafos y relaciones no euclidianas**. Proporciona una metodología formal para el modelado de mundo relacional y jerárquico, permitiendo que un modelo capture tanto dinámicas locales a nivel de nodo/motivo como propiedades topológicas globales.

## Enlaces Relacionados
- Paper: [[2026_HP-JEPA]]
- Arquitecturas Relacionadas: [[C-JEPA]], [[I-JEPA]], [[LeWorldModel]]

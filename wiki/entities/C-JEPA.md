---
title: "C-JEPA (Causal-JEPA)"
tags: [entity, architecture, jepa, object-centric, causal]
---

# C-JEPA (Causal-JEPA)

## Descripción
**C-JEPA** es una arquitectura World Model introducida en 2026 que combina la inferencia predictiva de la familia [[JEPA]] con la representación centrada en objetos (Object-Centric Representation, ej. Slot Attention) y sesgos inductivos causales.

## Arquitectura y Funcionamiento
1. **Object-Centric Encoder**: En lugar de extraer características de parches espaciales, C-JEPA extrae "slots" que representan entidades u objetos distintos en la escena.
2. **Object-Level Masking**: Durante el entrenamiento, se selecciona un objeto y se enmascara todo su historial (excepto un ancla de identidad inicial) y su futuro. 
3. **Auxiliary Variables**: Las acciones de control (ej. movimientos del brazo robótico) se tratan como entidades u objetos separados que condicionan al predictor.
4. **Predictor**: Un Transformer bidireccional que procesa el historial parcialmente enmascarado y las variables auxiliares para predecir los tokens futuros enmascarados.

## Diferencia Clave
La innovación de C-JEPA no es una nueva función de pérdida (utiliza una minimización estándar sobre representaciones latentes, similar a [[LeWM_Loss]] o VICReg adaptado a objetos), sino **qué es lo que se enmascara**. Enmascarar objetos enteros fuerza el aprendizaje relacional, mientras que enmascarar parches ([[V-JEPA2.1]]) fomenta la comprensión geométrica densa. 

Ambos enfoques son ortogonales pero revelan cómo el "masking" controla el sesgo inductivo del World Model.

## Enlaces Relacionados
- Paper: [[2026_Causal-JEPA]]

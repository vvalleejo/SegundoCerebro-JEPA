---
title: "Rectified LpJEPA"
tags: [entity, architecture, jepa, sparsity, rdmreg]
---

# Rectified LpJEPA

## Descripción
**Rectified LpJEPA** (2026) es un variante de arquitectura y función de pérdida para World Models de la familia JEPA que fuerza que el espacio de representación latente sea **esparzo (sparse)** y **no negativo**, en contraposición a las representaciones continuas y densas de modelos anteriores como [[LeWorldModel]].

## Componentes Matemáticos
- **RDMReg (Rectified Distribution Matching Regularization)**: Es el corazón de la arquitectura. Reemplaza al [[SIGReg]]. Mientras SIGReg alinea los features hacia una distribución Gaussiana, RDMReg los alinea hacia una distribución **RGG (Rectified Generalized Gaussian)**.
- **Ventaja**: Permite al investigador ajustar matemáticamente el trade-off entre la cantidad de "ceros" en la red (esparsidad) y la entropía de la información retenida.

## Conexión Neuronal
Esta variante acerca los World Models artificiales a los modelos biológicos (como la codificación predictiva en el córtex visual), donde las neuronas operan bajo estrictas restricciones metabólicas (activación esparza) y las tasas de disparo son inherentemente no negativas.

## Enlaces Relacionados
- Paper: [[2026_Rectified_LpJEPA]]

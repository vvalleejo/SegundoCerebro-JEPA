---
title: "LpWorldModel (LpWM)"
tags: [entity, architecture, world-models, sparse-representations, rdmreg]
---

# LpWorldModel (LpWM)

## Descripción
**LpWorldModel (LpWM)** es un modelo de mundo JEPA (NYU, AMI Labs, 2026) que explota representaciones latentes **dispersas y no negativas** para simplificar la predicción de dinámicas controladas. El modelo postula que las representaciones de entropía máxima o isotrópicas densas clásicas dificultan el descubrimiento de regularidades dinámicas, mientras que los códigos distribuidos dispersos "factorizan" el modelado del mundo.

## Componentes Clave
1. **Regularización RDMReg**: Utiliza Rectified Distribution Matching Regularization ([[RDMReg]]) en lugar de VICReg o [[SIGReg]] para hacer converger las estadísticas marginales de las características a una distribución de Laplace Rectificada, obteniendo representaciones estrictamente no negativas y dispersas.
2. **RepReLU (Reparameterized ReLU)**: Función de activación modificada que actúa como una ReLU exacta hacia adelante (generando verdaderos ceros), pero retiene la propagación de gradientes de un GeLU hacia atrás para evitar extinciones de neuronas.
3. **Dinámicas Factorizadas por Modos**:
   - El *soporte* binario (las neuronas que se encienden) sirve como un identificador que cambia ante transiciones de fase física (ej. contacto o colisión).
   - La *magnitud* de las características activas modela los estados continuos y locales como la posición y velocidad en dicho régimen de contacto.

## Implicaciones para World Models
- **Linealización Latente**: Demuestra una aproximación teórica donde representaciones *one-hot* de muy alta dimensión logran dinámicas de transición exactamente lineales.
- **Eficiencia Computacional en Predicción**: Mientras entornos complejos como *PushT* exigen arquitecturas de predictores gigantes (Transformers DiT / AdaLN) para funcionar sobre latentes densos, LpWM puede planificar exitosamente utilizando predictores de muchísima menor capacidad (MLPs variantes o invariantes en el tiempo) debido a la simplificación geométrica aportada por la dispersión.

## Enlaces Relacionados
- Paper: [[2026_LpWM]]
- Relacionado a la matemática estructural: [[Rectified_LpJEPA]]
- Arquitectura densa contraparte: [[LeWorldModel]]

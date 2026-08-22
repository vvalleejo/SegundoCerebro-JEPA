---
title: "Music-JEPA"
tags: [entity, architecture, jepa, audio, music, world-models, action-conditioned]
---

# Music-JEPA

## Descripción
**Music-JEPA** (2026, NYU / McGill / MBZUAI / AMI Labs) es un modelo de mundo para el dominio acústico-musical basado en [[JEPA]] que formula la música como un **sistema dinámico interactivo condicionado por acciones**, donde el audio representa el estado y los eventos de interpretación instrumental (pianoroll y pedal de resonancia) representan las acciones de control.

## Componentes Clave
1. **State & Action Tokenizers**: Encoders ViT para proyectar espectrogramas de audio ($s_t$) y matrices de notas/pedal ($a_t$) a representaciones latentes desacopladas.
2. **Action-Conditioned State Predictor**: Módulo con *cross-attention* capa por capa para predecir el siguiente estado latente $s_{t+1} = f(s_t, a_{t+1})$.
3. **Action Prior Dynamics**: Predictor complementario $g(a_t)$ que modela la estructura temporal de las transiciones de acordes y notas.
4. **Transcripción por Planificación Latente**: Infiere acciones musicales óptimas para reproducir un sonido objetivo resolviendo un problema inverso en el espacio latente vía optimización amortizada.

## Relevancia para el Doctorado
Music-JEPA ilustra la versatilidad de la arquitectura JEPA para formalizar dominios continuos y complejos como **sistemas de control activo**, extendiendo los modelos de mundo más allá de la visión física hacia el sonido estructurado y la cognición auditiva/motriz.

## Enlaces Relacionados
- Paper: [[2026_Music-JEPA]]
- Modelos Multimodales y Temporales: [[MJEPA]], [[CHARM]], [[V-JEPA2]], [[LeWorldModel]]

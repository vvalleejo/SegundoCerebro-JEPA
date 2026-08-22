---
title: "When Does LeJEPA Learn a World Model?"
authors: [David Klindt, Yann LeCun, Randall Balestriero (Cold Spring Harbor Lab, NYU, Brown)]
year: 2026
tags: [paper, lejepa, world-models, theory, identifiability, sigreg, math, lean4]
---

# When Does LeJEPA Learn a World Model?

## Resumen Ejecutivo
Este paper fundamental (2026) proporciona la **primera prueba matemática formal de identificabilidad para las arquitecturas JEPA**. Responde a una pregunta crucial para cualquier investigador en modelos de mundo: *¿Cuándo podemos garantizar matemáticamente que la representación latente aprendida por un JEPA recupera fielmente la estructura real del mundo?*

Demuestra que **LeJEPA** (combinación de pérdida de alineación y regularización gaussiana [[SIGReg]]) recupera las variables latentes verdaderas del mundo mediante una transformación puramente lineal e isométrica (hasta una rotación ortogonal $Q \in O(n)$), propiedad conocida como **Identificabilidad Lineal (Linear Identifiability)**.

## Resultados Teóricos Principales

### 1. Teorema 1: Identificabilidad Lineal de LeJEPA (Directo)
Si el mundo tiene variables latentes gaussianas independientes $z \sim \mathcal{N}(0, I_n)$ que evolucionan bajo un proceso de ruido aditivo estacionario (proceso Ornstein-Uhlenbeck $z' = \rho z + \sqrt{1-\rho^2}\eta$), la única representación que minimiza el objetivo de LeJEPA manteniendo marginales gaussianas es una rotación lineal de los latentes reales: $h(z) = Qz$.
- **Mecanismo de prueba**: Descomposición espectral mediante **Polinomios de Hermite** y la fórmula de Mehler. Se prueba que cualquier componente no lineal (grados de Hermite $d \ge 2$) se atenúa a un ritmo $\rho^d < \rho$, penalizando estrictamente cualquier no-linealidad en la alineación.

### 2. Teorema 2: Unicidad Gaussiana (Recíproco)
Entre todos los mundos estacionarios con ruido aditivo, la distribución **Gaussiana es la ÚNICA distribución latente** para la cual se cumple la identificabilidad lineal.
- **Mecanismo de prueba**: Teoría de Sturm-Liouville y análisis de la función de puntuación (*score function*). Si la primera autofunción no constante es afín, la función de puntuación debe ser lineal, lo que caracteriza únicamente a la Gaussiana.

### 3. Teorema 3: Identificabilidad Aproximada
En la práctica, la alineación y el blanqueamiento no son perfectos ($\delta, \varepsilon > 0$). El teorema demuestra que el error de recuperación de la representación se degrada de forma suave (*degrades gracefully*) en función del gap de alineación $\delta/(2\rho(1-\rho))$.

### 4. Teorema 4: Planificación Latente Óptima
Demuestra que si un codificador cumple la identificabilidad lineal ortogonal ($h(z) = Qz$), cualquier plan de acción optimizado en el espacio latente del modelo de mundo (ej. líneas rectas o MPC) es **matemáticamente idéntico al plan óptimo en el mundo real**, con las mismas acciones y el mismo coste.

## Verificación Formal
Todos los 5 resultados teóricos fueron verificados de manera computacional e irrefutable usando el demostrador formal de teoremas **Lean 4** (con la librería Mathlib).

## Implicaciones para el Doctorado
Este trabajo justifica teóricamente por qué las sondas lineales (*linear probes*) funcionan como evaluación estándar en SSL/JEPA: evaluar con sondas lineales solo es conceptualmente válido si la red ha aprendido una representación linealmente identificable. Además, explica por qué la regularización Gaussiana de [[SIGReg]] en [[LeWorldModel]] no es un truco heurístico, sino una condición matemáticamente necesaria y suficiente para garantizar que un World Model recupere los verdaderos grados de libertad del entorno.

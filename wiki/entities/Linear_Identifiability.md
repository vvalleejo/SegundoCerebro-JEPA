---
title: "Linear Identifiability in JEPAs"
tags: [entity, theory, jepa, identifiability, math]
---

# Linear Identifiability en JEPAs

## Descripción
La **Identificabilidad Lineal (Linear Identifiability)** es una propiedad matemática clave que garantiza que una representación aprendida mediante aprendizaje auto-supervisado (SSL) desentraña (*disentangles*) y recupera fielmente los verdaderos factores de variación o grados de libertad del mundo real ($z$), a menos de una transformación lineal/ortogonal simple: $h(z) = Qz$ con $Q \in O(n)$.

## Avance Teórico (Klindt, LeCun, Balestriero 2026)
Antes de este trabajo, el éxito de JEPAs como [[LeWorldModel]] o [[I-JEPA]] se atribuía empíricamente a la prevención del colapso y al foco en características lentas. Este trabajo demuestra formalmente que:

1. **Alineación + Regularización Gaussiana ([[SIGReg]])**: Fuerza al mapa aprendido a ser estrictamente lineal. Cualquier no-linealidad en la representación atenúa la correlación temporal entre pares positivos (demostrado a través de la descomposición en Polinomios de Hermite y la fórmula de Mehler).
2. **Unicidad de la Gaussiana**: La distribución Gaussiana es la *única* distribución objetivo dentro de procesos de ruido aditivo estacionarios para la cual se sostiene la identificabilidad lineal exacta.
3. **Garantía para la Planificación Latente**: Proporciona el respaldo teórico de que planificar en el espacio latente del modelo de mundo (ej. líneas rectas o MPC) produce trayectorias y decisiones equivalentes a las del mundo real sin distorsión geométrica.

## Enlaces Relacionados
- Paper: [[2026_LeJEPA_Identifiability]]
- Conceptos Matemáticos: [[SIGReg]], [[LeWM_Loss]]
- Arquitecturas Relacionadas: [[LeWorldModel]], [[Rectified_LpJEPA]]

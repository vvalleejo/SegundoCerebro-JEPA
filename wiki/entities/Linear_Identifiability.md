---
title: "Linear Identifiability in JEPAs"
type: "entity"
tags: [entity, theory, jepa, identifiability, math]
---

# Linear Identifiability en JEPAs

## Descripción
La **Identificabilidad Lineal (Linear Identifiability)** es la propiedad de que una representación aprendida mediante aprendizaje auto-supervisado (SSL) recupere los verdaderos factores latentes del mundo ($z$) **salvo una rotación ortogonal**: $h(z) = Qz$ con $Q \in O(n)$.

> [!warning] No es *disentanglement*
> Una rotación $Q$ mezcla las coordenadas, así que cada eje de $h$ no corresponde a un único factor. En el paper, "disentangled" describe los latentes del **mundo**, no la representación aprendida (Fig. 2, p. 3). La identificabilidad lineal es además condición *"necessary, albeit not sufficient"* para que un linear probe sea fiel (p. 2).

## Avance Teórico (Klindt, LeCun, Balestriero 2026)
Antes de este trabajo, el éxito de JEPAs como [[LeWorldModel]] o [[I-JEPA]] se atribuía empíricamente a la prevención del colapso y al foco en características lentas. Este trabajo demuestra formalmente que:

1. **Alineación + Regularización Gaussiana ([[SIGReg]])** (Thm. 1): si los latentes del mundo son Gaussianos independientes $z\sim\mathcal N(0,I_n)$ con transición Ornstein–Uhlenbeck $z'=\rho z+\sqrt{1-\rho^2}\eta$ y $m=n$, el **óptimo global** del objetivo LeJEPA (alineación $L_2^2$ + marginal $\mathcal N(0,I)$) es $h(z)=Qz$. Cualquier componente no lineal atenúa la correlación entre pares positivos (descomposición en polinomios de Hermite y fórmula de Mehler).
2. **Unicidad de la Gaussiana** (Thm. 2): entre los mundos estacionarios con ruido aditivo, la Gaussiana es la *única* distribución **de los latentes del mundo** para la que se cumple la identificabilidad lineal. Fuera de ella falla; en trayectorias RL reales el $R^2$ total "never exceeds 0.5" (Tab. 2).
3. **Planificación latente** (Thm. 4): si $h(z)=Qz$ y la dinámica empujada es exacta, el plan óptimo en latente coincide con el del mundo real. Exige costes **invariantes bajo $O(n)$** (Eq. 6): $L_2$ sí; $L_1$ o Huber elemento a elemento, no.

> [!important] Alcance
> - Es un resultado de **población** sobre el óptimo global. No dice nada de muestras finitas ni de dinámica de entrenamiento (§7).
> - Cubre el **encoder**, no la dinámica condicionada por acciones (App. D.2: *"our theorems do not prove that it is"*).
> - No incluye EMA, stop-gradient ni predictor. No establece umbrales de $R^2$ para probes.
> - VICReg e InfoNCE también alcanzan identificabilidad en los experimentos (Tab. 1), así que no es exclusivo de SIGReg.

## Enlaces Relacionados
- Paper: [[2026_LeJEPA_Identifiability]]
- Conceptos Matemáticos: [[SIGReg]], [[LeWM_Loss]]
- Arquitecturas Relacionadas: [[LeWorldModel]], [[Rectified_LpJEPA]]
- **Arquitectura**: [[wiki/architecture/2026_LeJEPA_Identifiability|2026_LeJEPA_Identifiability]]

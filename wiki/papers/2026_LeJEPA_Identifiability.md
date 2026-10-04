---
title: "When Does LeJEPA Learn a World Model?"
authors: [David Klindt, Yann LeCun, Randall Balestriero (Cold Spring Harbor Lab, NYU, Brown)]
year: 2026
venue: "arXiv preprint"
arxiv: "2605.26379"
source_pdf: "raw/When Does LeJEPA Learn a World Model.pdf"
repo: "https://github.com/klindtlab/lejepa-identifiability"
type: "paper"
family: "jepa"
modality: [theory]
anti_collapse: [sigreg]
predictor: false
planner: [none]
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
Los **5 teoremas** (los Thm. 1–4 anteriores más el **Thm. 5** del App. E, que prueba la identificabilidad vía energía de Dirichlet con ruido infinitesimal y Mazur–Ulam) están verificados en **Lean 4** con Mathlib v4.28.0 y "zero sorry obligations" (App. G, p. 31). Matiz importante: la verificación es "modulo standard background lemmas axiomatized from the literature" (p. 5). Se toman como axiomas la completitud de Hermite/Mehler, Mazur–Ulam, AM-GM, Jensen y el pushforward de trayectorias, entre otros (Tab. 4, p. 38). Lo que se verifica son las cadenas de razonamiento entre esos axiomas.

> [!important] Condiciones que suelen omitirse
> - **Theorem 4** (planificación) exige que los costes de etapa y terminal sean **invariantes bajo $O(n)$**: $\ell(Rz,a)=\ell(z,a)$ para todo $R\in O(n)$ (Eq. 6). El coste $L_2$ de LeWM lo cumple; la energía $L_1$ de V-JEPA 2 **no** es invariante por rotación (observación de la bóveda, no del paper).
> - El resultado cubre el **encoder**, no la dinámica condicionada por acciones (App. D.2: "our theorems do not prove that it is").
> - Se asume dimensión de salida igual a la latente real ($m=n$, p. 9). No se enuncia inyectividad de $g$: los teoremas son sobre $h=f\circ g$ medible.

## Implicaciones para el Doctorado
El paper da una **motivación teórica** a las sondas lineales (*linear probes*) en SSL/JEPA, no una justificación general. La identificabilidad lineal es condición *"necessary, albeit not sufficient"* para un probing lineal fiel (p. 2). Un $R^2$ alto en probes es evidencia empírica compatible con ella, pero no la demuestra.

Sobre [[SIGReg]] en [[LeWorldModel]]: el paper muestra que, **bajo sus hipótesis** (latentes Gaussianos con transición OU, $m=n$, óptimo global), el objetivo LeJEPA recupera los latentes salvo rotación. Eso da un respaldo teórico a SIGReg frente a otras heurísticas. Sin embargo, **no** prueba que SIGReg sea condición necesaria y suficiente:
- El Thm. 2 (unicidad) trata de la distribución de los **latentes del mundo**, no del regularizador.
- VICReg e InfoNCE también alcanzan identificabilidad en la Tab. 1 (p. 9).
- Mundos no Gaussianos, dinámica condicionada por acciones, EMA o pérdidas distintas de $L_2^2$ quedan fuera del resultado.

## Referencias Cruzadas
- **Arquitectura**: [[wiki/architecture/2026_LeJEPA_Identifiability|2026_LeJEPA_Identifiability]]

---
title: "Deep Learning-Based Artificial Intelligence for Predictive Maintenance in Smart Manufacturing Systems"
authors: [Yann LeCun]
year: 2026
venue: "Sin venue (manuscrito PDF sin arXiv ni DOI)"
arxiv: ""
source_pdf: "raw/Deep Learning-Based Artificial Intelligence for Predictive.pdf"
repo: ""
type: paper
family: review
modality: [time-series]
anti_collapse: [n/a]
predictor: false
planner: [none]
work_type: revisión-conceptual
provenance: dudosa
tags: [paper, review, predictive-maintenance, time-series, industrial-iot, rul]
---

# Deep Learning-Based AI for Predictive Maintenance in Smart Manufacturing Systems

> [!warning] Procedencia: dudosa, no usar como fuente primaria
> - **Autoría atribuida:** "Yann LeCun, New York University and AMI Labs" como **autor único** (PDF p. 1). Es un trabajo fuera de su línea de investigación y no tiene arXiv, DOI ni venue.
> - **Metadatos del PDF:** `author: python-docx`, `creator: Microsoft Word 2021`, creado el 2026-08-25. Todo apunta a que se generó por script y no con el pipeline LaTeX habitual de sus papers.
> - **Bibliografía con relleno ajeno al tema** (refs. 36–43, pp. 8–9): ataques 5G/6G, malware, "The SpaceX IPO…", la estrategia de IA de Ghana (con una fecha incoherente, "(2025-2035)… (2016)") y fraude cibernético en Canadá. Es el patrón típico de *citation padding*.
> - **Sin resultados propios, según el propio texto:** "No new experiment, dataset, model training, or performance result is claimed" (§1, p. 2). Tampoco hay recuento PRISMA (§3, p. 3).
> - **No contiene nada de JEPA ni de world models.** Se ingiere para que la bóveda esté completa y como **contexto aplicado** (series temporales industriales), sin peso en las síntesis sobre JEPA.
>
> *Juicio de la bóveda:* es muy probable que la atribución a LeCun sea falsa o errónea. No es verificable con el PDF.

## Resumen ejecutivo

Es una revisión crítica, dirigida y **no sistemática**, sobre deep learning para mantenimiento predictivo (PdM) en fabricación inteligente (§3, p. 3). Propone un **framework conceptual** de 11 capas (Fig. 2, p. 5), que va de la captura IIoT a la monitorización continua del modelo. Está "literature-derived, and not experimentally validated" (Abstract; §5).

La tesis central es que un PdM accionable requiere un **sistema end-to-end**: calidad del dato, definición de la tarea, generalización, incertidumbre, fidelidad de las explicaciones y supervisión humana. Un clasificador aislado no basta (§7).

## Contenido

- **Taxonomía de tareas** (§2): la revisión insiste en que son analíticamente distintas y que confundirlas lleva a modelos mal especificados.
  - *fault detection*
  - *fault diagnosis*
  - *health assessment*
  - *failure prediction*
  - estimación de *RUL*
  - *decision support*
- **Arquitecturas frente a tareas** (Tab. 1, pp. 2–3):
  - **CNN:** morfología local de la señal y espectrogramas.
  - **LSTM/GRU:** degradación secuencial y RUL.
  - **Autoencoder:** anomalías con pocas etiquetas. La revisión advierte que el error de reconstrucción indica desviación, no la causa del fallo (§4.2).
  - **CNN-LSTM:** combina las dos anteriores.
  - **Transformer:** dependencias largas. Los pesos de atención no son explicaciones (§4.2).
  - **Transfer learning:** riesgo de *negative transfer*.
  - **SSL:** "particularly promising" por la abundancia de datos industriales sin etiquetar, aunque exige validación downstream (§4.2).
  - **GNN:** solo si el grafo físico está justificado.
- **Estrategias de mantenimiento** (Tab. 2, p. 4): correctivo, preventivo, basado en condición y PdM guiado por IA.
- **RUL e incertidumbre** (§4.3): una estimación puntual de RUL sin intervalo "may be operationally misleading".
- **Despliegue** (§4.4, §6): umbrales según criticidad y coste, XAI (SHAP, LIME, saliency, contrafactuales) con fidelidad evaluada aparte, edge frente a cloud, ciberseguridad, *concept drift* y riesgos del aprendizaje continuo.

## Arquitectura

Ver [[wiki/architecture/2026_DL_PdM_Framework_Arch|arquitectura del framework PdM]].

## Formulación matemática

No tiene. El documento no contiene ecuaciones.

## Resultados clave

No tiene resultados cuantitativos (§3).

## Relevancia para la bóveda

- **Conexión temática, no metodológica,** con [[CHARM]] (JEPA para series temporales multivariantes de sensores): el problema que la revisión atribuye a los datos industriales (muchos datos sin etiquetar, pocos fallos) es exactamente el escenario en que el SSL de tipo JEPA tiene sentido.
- **Las limitaciones que enumera** (cambio de régimen, deriva, incertidumbre del RUL) son preguntas abiertas para los world models latentes aplicados a señales físicas. Ver §11 de [[JEPA-master-note]].

## Referencias cruzadas

- **Arquitectura:** [[wiki/architecture/2026_DL_PdM_Framework_Arch|2026_DL_PdM_Framework_Arch]]
- **Relacionados:** [[CHARM]], [[JEPA-master-note]]

---
title: LeJEPA
tags: [architecture, entity, jepa]
---

# LeJEPA

LeJEPA (presentado por Balestriero y LeCun, 2025) es el pilar fundamental que introduce un aprendizaje predictivo autosupervisado escalable y comprobablemente libre de colapso, pero **sin las heurísticas** clásicas de modelos Joint-Embedding anteriores (como redes target con promedios móviles, stop gradients o proyectores/predictores asimétricos).

Logra esto reemplazando esas asimetrías arquitectónicas por una restricción explícita en la distribución latente conocida como [[SIGReg]]. En 2026, fue adaptada con éxito al dominio del video a través de la arquitectura [[LeVJEPA]].

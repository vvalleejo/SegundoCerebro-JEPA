---
title: "AD-E2E-JEPA (world model JEPA para conducción autónoma end-to-end)"
type: entity
tags: [entity, jepa, world-models, autonomous-driving, sigreg, zero-shot-planning]
---

# AD-E2E-JEPA

**Qué es:** un world model JEPA acción-condicionado para conducción autónoma end-to-end (*Autonomous Driving with an End-to-End JEPA*), de NYU / AMI Labs (Zhu, Zhang, LeCun, Choromanska; arXiv 2026). Toma como base [[JEPA-WM]] (DINOv3 ViT-L congelado + predictor AdaLN/RoPE) e inserta un **patch projector** convolucional entrenable que comprime los parches 16× en número y 4× en dimensión. El proyector se regulariza con stop-gradient + [[SIGReg]] parche a parche.

## Ideas clave
1. **Eficiencia sin perder calidad:** 0.8 s por escena frente a 91.8 s (DINO-WM) y 101.0 s (JEPA-WM), con EPDMS 76.6 frente a 68.3 / 74.2 en 100 escenas (Tab. 2).
2. **Planificación por vocabulario:** en lugar de CEM, evalúa en paralelo un subconjunto del vocabulario de 8192 trayectorias de VADv2 y elige la de menor distancia latente al frame objetivo (Eq. 13–14).
3. **Evaluación del world model con objetivo oráculo:** el objetivo es el frame futuro real. Mide la calidad del modelo (FDE, hit rate), no la capacidad de conducir de forma autónoma.
4. **Transferencia:** el proyector preentrenado sube un modelo de imitation learning de 80.2 a 85.4 EPDMS (Tab. 3).

## Resultado principal
En el test completo de NAVSIMv2 (12 146 escenas): EPDMS 67.3 / EPDMS† 84.1 con 256 candidatas, y 72.9 / 86.5 con 8192 (FDE 2.8 m, $\Delta\theta$ 2.0°) (Tab. 2).

## Enlaces
- Paper: [[wiki/papers/2026_AD-E2E-JEPA|2026_AD-E2E-JEPA]]
- Arquitectura: [[wiki/architecture/2026_AD-E2E-JEPA_Arch|2026_AD-E2E-JEPA_Arch]]
- Matemáticas: [[SIGReg]], [[Teacher_Forcing_Rollout_Loss]]
- Relacionados: [[JEPA-WM]] (base), [[DINO-WM]] y [[LeWorldModel]] (baselines), [[Temporal_Straightening]] (origen del proyector), [[JEPA]]

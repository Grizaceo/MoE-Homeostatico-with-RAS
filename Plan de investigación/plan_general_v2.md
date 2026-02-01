# Plan general v2 (prioridad: espacios mixtos + RAS editor de routing/folding)

## Principio rector
Construir primero un *baseline* reproducible (Euclídeo + Ricci-cleaning + routing), y luego **subir complejidad** con un *Geometric MoE* (producto/mezcla de variedades) donde el **RAS** (como meta-control) pueda:
1) asignar nodos/subgrafos al espacio geométrico más útil,
2) editar *routing* (selección de trayectorias) y *folding* (rewiring/atajos),
3) controlar presupuesto (homeostasis) y disparar verificación (anti-alucinación).

## Estructura de artefactos (mínimos)
- `paso1_baseline_ricci_v2.md`: baseline + ricci cleaning + routing métrico.
- `paso2_multiespacio_ras_v2.md`: producto/mezcla de variedades + gating RAS + folding.
- `paso3_benchmark_v2.md`: bancos de prueba sintéticos + métricas + tablero de resultados.
- `paso4_verificacion_v2.md`: política de verificación estratificada RSI + bucles RAS.

## Convención RSI (operativa)
- **Real**: rupturas/alertas duras (inestabilidad numérica, latencia p95, pérdida de conectividad).
- **Simbólico**: contratos, verificación, reglas de enrutamiento, auditoría de evidencia.
- **Imaginario**: salida generativa, narrativa, UX; se mantiene “libre” pero bajo límites S/R.

## Criterio de avance (gates)
No se pasa de un paso al siguiente si no existe:
- dataset/escenarios definidos,
- métrica principal con umbral,
- registro de evidencia,
- prueba mínima automatizable,
- y reporte comparativo vs baseline.

## Próximos “deliverables” (para iniciar)
1) Tabla de métricas y escenarios (Paso 3) definida antes de complicar geometría.
2) Baseline reproducible con Ricci-cleaning (Paso 1).
3) Primer Geometric MoE (Paso 2) con 2 espacios (E + H), antes de añadir S/torus/etc.
4) Loop RAS→(routing/folding/verificación) (Paso 4) con logs trazables.

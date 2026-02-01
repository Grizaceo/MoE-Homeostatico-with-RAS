# Paso 4 v2 — Política de verificación RSI + bucle RAS (anti-alucinación + control)

## Objetivo
Hacer que el sistema:
- no emita *claims* factuales sin soporte,
- registre evidencia por claim,
- y use el RAS para decidir cuándo verificar, cuándo escalar expertos, y cuándo degradar (modo frugal).

## Clasificación RSI de fallas (operativa)
- **Real**: ruptura dura (inconsistencia lógica detectada, desconexión del grafo, NaNs, latencia p95 > umbral).
- **Simbólico**: violación de contrato (sin fuente, regla rota, evidencia incompleta).
- **Imaginario**: fluidez/estilo sin anclaje (texto plausible pero no verificado).

## Política mínima (por claim)
1) Si el claim es factual y “carga” la decisión: exigir al menos 1 verificación:
   - RAG/documento, o
   - verificador interno (2ª pasada), o
   - consistencia cruzada (n>1 muestras + ranking).
2) Si no se puede verificar: marcar como incierto + proponer test.

## Acciones RAS (gating)
- Escalar: activar verificación + más cómputo (ACT) + más espacios (mixtura).
- De-escalar: recortar cómputo y reducir variedad geométrica si hay stress.
- Re-estructurar: disparar Ricci-cleaning o folding (controlado) si la topología se degrada.

## Deliverables
- Esquema de logs:
  - `claim_id`, `rsi_tag`, `evidence_ref`, `confidence`, `ras_action`, `cost`.
- Checklist automático pre-respuesta (modo agente en chat):
  - ¿qué es claim factual?
  - ¿qué evidencia lo soporta?
  - ¿qué acción RAS corresponde?

## Gate de aceptación
- % claims con evidencia válida ≥ umbral (p.ej. 80% en tareas factuales).
- Tasa de contradicción interna bajo X.
- Degradación controlada bajo stress (curva calidad vs costo).

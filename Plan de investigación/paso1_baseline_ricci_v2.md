# Paso 1 v2 — Baseline (Euclídeo) + Ricci-cleaning + Routing

## Objetivo
Tener un punto de referencia estable y reproducible para:
- medir ganancia real de curvatura/mixturas,
- aislar el efecto de Ricci-cleaning (denoising estructural),
- obtener métricas base de routing (éxito/latencia/distorsión).

## Inputs
- Grafo(s): sintéticos (árbol, SBM, grid) y/o un dataset estándar (opcional).
- Parámetros de ruido: prob. de rewiring, adición/eliminación de aristas, “heterofilia” artificial.
- Definición de tarea de routing: greedy routing o shortest-path como oracle.

## Operación central
1) Calcular curvatura Forman–Ricci por arista.
2) Aplicar **Ricci-cleaning**: remover o atenuar aristas con curvatura “mala” (umbral).
3) Ejecutar queries de routing antes/después.
4) Registrar cambios topológicos (componentes, clustering, distancia media).

## Deliverables
- Script/notebook reproducible (o pseudo-código si aún no hay repo).
- Log estructurado de métricas por corrida.
- Informe comparativo: *before/after* Ricci-cleaning.

## Métricas mínimas
- Éxito de routing (% de entregas o rutas correctas).
- Longitud media de ruta / stretch vs oracle.
- Conectividad residual (#componentes, aristas removidas).
- Latencia (p50/p95) de cálculo de curvatura + limpieza.
- Estabilidad: varianza por semilla.

## Riesgos
- Umbral de “limpieza” rompe conectividad (Real): mitigar con constraint de conectividad mínima.
- Sobre-limpieza = pérdida de atajos útiles: mitigar con validación por benchmark.

## Prueba mínima de aceptación (gate)
- Mejora ≥ X% en éxito de routing o reducción ≥ Y% en stretch
- sin caer por debajo de Z% de conectividad (o sin aumentar #componentes).

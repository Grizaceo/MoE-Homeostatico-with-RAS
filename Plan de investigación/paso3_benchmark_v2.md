# Paso 3 v2 — Benchmark y tablero comparativo (obligatorio antes de escalar)

## Objetivo
Definir benchmarks mínimos (sintéticos) y métricas para comparar:
- baseline vs mixturas,
- con/ sin Ricci-cleaning,
- con/ sin folding,
- con/ sin RAS gating.

## Escenarios mínimos (3 grafos)
1) Árbol jerárquico (crecimiento exponencial).
2) SBM (comunidades) con ruido.
3) Grid 2D (estructura local) con rewiring aleatorio.

## Tareas
- Routing: greedy vs oracle.
- Link prediction (opcional) para medir coherencia estructural.
- Robustez al ruido: barrido de p_noise.

## Métricas (mínimas)
### Veracidad/Coherencia del sistema
- Contradicción interna (definir test de inconsistencias topológicas o rutas incompatibles).
- Cobertura de evidencia (si se integra verificador; ver Paso 4).

### Desempeño operativo
- Latencia p50/p95 por componente: curvatura, embedding, gating, folding.
- Costo (tokens/llamadas) si hay LLM en loop.

### Estabilidad homeostática
- Eventos de estrés (umbral de latencia, desconexión, inestabilidad numérica).
- Curva calidad vs costo bajo stress.

## Deliverables
- “Matriz de experimentos” (tabla): factores × niveles.
- Plantilla de reporte (markdown) para resultados.
- Umbrales (gates) para permitir avanzar.

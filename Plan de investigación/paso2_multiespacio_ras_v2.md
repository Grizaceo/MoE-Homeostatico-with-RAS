# Paso 2 v2 — Espacios mixtos (producto/mezcla de variedades) + RAS editor de routing/folding

## Objetivo
Introducir un **Geometric MoE**: cada nodo/subgrafo puede representarse en múltiples espacios (p.ej. Euclídeo + Hiperbólico + Esférico),
y un **gating** que asigna (soft/hard) a qué espacio(s) vive la representación y por dónde se rutea.

El RAS actúa como *meta-control*:
- decide cuándo aumentar complejidad geométrica,
- ajusta pesos/curvaturas del gating,
- ejecuta folding (rewiring) condicionado a métricas,
- y dispara verificación cuando el riesgo de alucinación aumenta.

## Modelo conceptual
### 1) Espacio producto / mezcla
- Producto: x = [x_E || x_H || x_S] (concatenación de coordenadas por variedad).
- Mezcla (MoE): x = Σ_k π_k(x) · emb_k(x) con π_k aprendible (softmax).

### 2) Señales para el gating (RAS)
- Entropía local / incertidumbre.
- Curvatura local (antes/después de Ricci-cleaning).
- Historial de éxito de routing (por región/subgrafo).
- Presupuesto homeostático (latencia/costo).

### 3) Folding como operador controlable
- Folding = añadir/remover aristas “atajo” según distancia geodésica en el espacio elegido.
- Reglas: 
  - solo permitir folding si no degrada conectividad / no aumenta contradicciones,
  - penalizar “overfolding” (ciclo vicioso de shortcuts).

## Deliverables
- Especificación de interfaces (mínimas):
  - `embed(space_id, node_features) -> coords`
  - `distance(space_id, coords_i, coords_j) -> d_ij`
  - `gate(node_state) -> π_k`
  - `fold(graph, coords, budget) -> graph'`
- Plan de ablation:
  1) E solo
  2) E + H (producto)
  3) E + H (mezcla MoE)
  4) (opcional) + S

## Métricas específicas
- Distorsión de embedding (si se define oracle de distancias).
- Ganancia marginal por espacio (Δ éxito routing / Δ latencia).
- Overhead del gating (ms, params, FLOPs).
- Estabilidad numérica (especialmente H).

## Gate de aceptación
- El modo mixto debe superar baseline (Paso 1) en éxito routing y/o distorsión
- con overhead controlado (< umbral de latencia p95).

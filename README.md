# MoE Homeostático con RAS

Sistema de Mixture of Experts (MoE) Homeostático con enfoque de Geometric Deep Learning.

## Arquitectura

El sistema implementa:
- **Curvatura Forman-Ricci**: Para análisis estructural de grafos
- **Ricci-Cleaning**: Limpieza basada en curvatura
- **Embeddings Multi-Espacio**: Euclídeo (MDS, Laplacian) + Hiperbólico (Poincaré)
- **MoE Gating**: Selección dinámica de espacio por nodo
- **Verificación RSI**: Logging Real-Simbólico-Imaginario

## Estructura

```
src/
├── core/               # Curvatura, limpieza, routing
│   ├── graph_utils.py
│   ├── curvature.py
│   ├── ricci_cleaning.py
│   └── routing.py
├── geometry/           # Embeddings multi-espacio
│   ├── euclidean.py
│   ├── hyperbolic.py
│   └── moe_gating.py
└── verification/       # Sistema RSI
    └── rsi_logger.py

tests/                  # Suite de tests
examples/               # Demos y scripts de validación
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

## Ejecutar Tests

```bash
python -m pytest -q
```

## Demos

```bash
# Fase 1: Ricci-Cleaning
python examples/demo_phase1.py

# Fase 2: Multi-Espacio + MoE
python examples/demo_phase2.py

# Validación pre-escalamiento
python examples/validate_adjustments.py
```

## Estado Actual

- [x] Fase 0: Scaffold
- [x] Fase 1: Baseline Ricci + Routing
- [x] Fase 2: Multi-Espacio + MoE Gating
- [x] Fase 2.5: Ajustes Pre-Escalamiento
- [ ] Fase 3: Benchmark + Dashboard
- [ ] Fase 4: RAS Bucles de Control

## Licencia

MIT

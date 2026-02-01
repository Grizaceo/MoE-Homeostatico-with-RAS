# PENDIENTE: Capa de Expertos Especializados con RAG

**Fecha:** 1 de Febrero, 2026  
**Estatus:** Diseño pendiente  
**Contexto:** Clarificación tras implementación del RSA Engine core

---

## Brecha Identificada

### Lo Implementado (RSA Engine Core)
- ✅ `RSASolver`: Motor de agregación recursiva
- ✅ `stratified_sample()`: Selección por similitud semántica
- ✅ `RepechageBuffer`: Rescate de outliers con curvatura estable
- ✅ `RASController`: Decisor de presupuesto N/K/T

### Lo que Falta (Expertos Persistentes)
- ❌ **Expertos pre-definidos** con especialización fija (código, leyes, etc.)
- ❌ **Modelos por experto** (potencialmente fine-tuned o LoRA)
- ❌ **RAG por experto** (base de datos personal/biblioteca)
- ❌ **Router** que decida qué expertos consultar según query

---

## Arquitectura Target

```
                     ┌─────────────────┐
                     │   Query Input   │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  RAS Controller │ ← Decide qué expertos + presupuesto
                     └────────┬────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │ Experto     │     │ Experto     │     │ Experto     │
  │ Código      │     │ Leyes       │     │ General     │
  ├─────────────┤     ├─────────────┤     ├─────────────┤
  │ Qwen-1.5B   │     │ Qwen-1.5B   │     │ Qwen-4B     │
  │ + LoRA code │     │ + LoRA law  │     │ (vanilla)   │
  ├─────────────┤     ├─────────────┤     ├─────────────┤
  │ RAG: repos, │     │ RAG: códigos│     │ Sin RAG     │
  │ docs API    │     │ jurisprud.  │     │             │
  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                     ┌────────▼────────┐
                     │ RSA Aggregator  │ ← Motor ya implementado
                     │ (estratifica y  │
                     │  agrega)        │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  Final Answer   │
                     └─────────────────┘
```

---

## Componentes Pendientes

### 1. Registro de Expertos (`src/experts/registry.py`)
```python
@dataclass
class ExpertConfig:
    name: str                    # "legal", "code", "general"
    model_path: str              # Ruta al modelo o adaptador LoRA
    embedding_model: str         # Para RAG
    vector_store_path: str       # ChromaDB/FAISS
    specialization: list[str]   # Tags de especialidad
    priority: int               # Orden de consulta
```

### 2. Expert Agent (`src/experts/agent.py`)
```python
class ExpertAgent:
    def __init__(self, config: ExpertConfig): ...
    def retrieve(self, query: str, k: int) -> list[Document]: ...
    def generate(self, query: str, context: list[Document]) -> str: ...
```

### 3. Router (`src/experts/router.py`)
```python
class ExpertRouter:
    def select_experts(self, query: str, max_experts: int) -> list[ExpertAgent]:
        """Decide qué expertos son relevantes para este query."""
        ...
```

### 4. Integración con RSA
- El `RSASolver` recibiría respuestas de **expertos reales** en lugar de un solo LLM
- Cada experto genera N/num_experts candidatos
- RSA agrega entre expertos

---

## Estimación de Recursos (RTX 4060 8GB)

| Configuración | Expertos | Modelo cada uno | VRAM estimada |
|---------------|----------|-----------------|---------------|
| Mínima | 2 | Qwen-1.5B Q4 | ~3GB |
| Recomendada | 3 | Qwen-1.5B Q4 | ~4.5GB |
| Máxima | 4 | Qwen-1.5B Q4 | ~6GB |

**Nota:** Con modelos cargados dinámicamente (load/unload), podrías tener más expertos pero solo 1-2 activos a la vez.

---

## Prioridad de Implementación

1. **P0 - LLM Adapter real**: Conectar Qwen via `llama.cpp` o `vLLM`
2. **P1 - RAG básico**: Un solo vector store compartido
3. **P2 - Multi-experto**: Registro + Router + RAG personalizado
4. **P3 - Fine-tuning/LoRA**: Entrenar expertos especializados

---

## Referencias

- Paper RSA: Venkatraman et al. (2026) - Recursive Self-Aggregation
- MoE Architecture: Mixture of Experts
- RAG: Retrieval Augmented Generation

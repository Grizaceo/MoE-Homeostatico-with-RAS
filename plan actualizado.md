# PROYECTO: MoE-Homeostatico-L-kn (Arquitectura RSA)

**Plataforma de Desarrollo:** Antigravity AI Agents (Google)
**Hardware Target:** Local RTX 4060 (8GB VRAM)
**Modelo Base:** Qwen 2.5 3B Instruct (Int4/Int8)

---

## 1. Visión del Proyecto
Este proyecto busca implementar capacidades de razonamiento profundo ("System 2 Thinking") en hardware de consumo limitado. A diferencia de los enfoques tradicionales que escalan el tamaño del modelo (Parámetros), nosotros escalamos el tiempo de inferencia y la recursividad.

El sistema se inspira en el modelo topológico RSI (Real-Simbólico-Imaginario) para mantener la coherencia, pero se implementa bajo ingeniería de software estricta.

## 2. Componentes Principales

### A. El Gerente: `L_kn` (Lacanian Knot Manager)
Anteriormente conocido como RAS. Es un agente ligero (o heurística) encargado de la **Homeostasis Metabólica**.
* **Función:** Evalúa la complejidad del query de entrada y el estado del hardware (`ComputeConstraints`).
* **Output:** Asigna presupuesto computacional dinámico: Población ($N$), Agregación ($K$) y Pasos ($T$).
* **Filosofía:** "No gastar energía de maratón para ir a la esquina".

### B. El Motor: RSA (Recursive Self-Aggregation)
Implementación adaptada del paper de Venkatraman et al. (2026).
* **Estrategia "Single-Backbone":** Utilizamos un único modelo cargado en VRAM que rota sus "máscaras" (System Prompts) para simular un comité de expertos.
* **Innovación - Estratificación:** No usamos muestreo aleatorio. Agrupamos respuestas semánticamente (`sentence-transformers`) para limpiar el ruido del contexto del experto.
* **Innovación - Repechaje (The Rescue):** Utilizamos Curvatura de Ricci (`GraphRicciCurvature`) para identificar respuestas "Outliers" (semánticamente distantes) pero "Robustas" (topológicamente consistentes) y reinsertarlas en el proceso para fomentar la creatividad.

---

## 3. Glosario de Refactorización (RSI -> Tech)
Para evitar alucinaciones filosóficas de los LLMs, hemos traducido los conceptos:

| Concepto Original (Lacan) | Concepto Técnico (Código) | Descripción |
| :--- | :--- | :--- |
| **RSI (Nudo Borromeo)** | **System Topology** | La estructura que mantiene unido el grafo. |
| **Real (R)** | **ComputeConstraints** | Límites duros: VRAM, Tiempo, Latencia. |
| **Simbólico (S)** | **LogicalConsistency** | Reglas sintácticas, contratos de API, nudos "locked". |
| **Imaginario (I)** | **PopulationVariance** | Diversidad de respuestas, entropía de muestreo. |
| **RAS** | **L_kn (Manager)** | El nodo de control y decisión. |

---

## 4. Dependencias Técnicas
Para reproducir este entorno en la plataforma Antigravity o localmente:

* `vllm` (Preferencia) o `llama-cpp-python`: Backend de inferencia.
* `sentence-transformers`: Para la vectorización ligera y estratificación.
* `networkx`: Gestión de grafos.
* `GraphRicciCurvature`: Cálculo de topología para el Repechaje.
* `numpy`, `scipy`, `pydantic`.

---

## 5. Apéndice: Escalabilidad y Heterogeneidad

**Nota sobre la implementación actual vs. ideal:**
La implementación actual utiliza **"Diversidad Sintética"** (Single-Backbone) debido a las restricciones de memoria de la RTX 4060. Un solo modelo simula diversidad mediante temperatura y prompts.

En un entorno de producción con hardware ilimitado (e.g., Clúster de H100s o múltiples 4090s), la arquitectura ideal migraría a **"Diversidad Heterogénea"**. En ese escenario, cada nodo del proceso RSA sería servido por modelos fundacionales distintos (Qwen, Llama, Mistral, DeepSeek) para lograr un **Desacoplamiento de Sesgos (Bias Decoupling)** real. La arquitectura `L_kn` está diseñada para ser agnóstica a este cambio: el "Gerente" puede orquestar tanto un solo modelo esquizofrénico como un clúster de modelos distintos.
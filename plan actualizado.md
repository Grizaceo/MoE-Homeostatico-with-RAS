# ANTIGRAVITY: Arquitectura de Bio-Orquestación Evolutiva (v2.0)

**Fecha:** 1 de Febrero, 2026
**Estatus:** Fase de Implementación Core
**Autor:** Grizaceo & Gemini (AI Partner)

---

## 1. Resumen Ejecutivo
El proyecto Antigravity ha evolucionado de un sistema de enrutamiento estático a una **Arquitectura de Agentes Híbrida**. El objetivo es resolver el trilema de **Coherencia, Creatividad y Costo** en sistemas de IA locales.

La solución adoptada separa la toma de decisiones ("Gerencia") de la ejecución del pensamiento ("Motor"), introduciendo mecanismos de **Justicia Algorítmica (Repechaje)** para evitar la convergencia prematura en soluciones mediocres.

## 2. Definición de Conceptos Clave

Hemos desambiguado las siglas críticas del proyecto:

### RAS (Reticular Activating System) - "El Gerente"
* **Rol:** Homeostasis y Economía de la Atención.
* **Función:** No resuelve el problema. Analiza el problema para asignar presupuesto.
* **Decisión:** Determina dinámicamente los hiperparámetros $N$ (Población), $K$ (Agregación) y $T$ (Tiempo) basándose en la energía disponible y la complejidad percibida.

### RSA (Recursive Self-Aggregation) - "El Motor"
* **Rol:** Imaginación y Refinamiento.
* **Función:** Genera poblaciones de ideas paralelas y las refina recursivamente (basado en Venkatraman et al., 2026).
* **Innovación Antigravity:** A diferencia del paper original, no usa muestreo aleatorio puro.

## 3. La Innovación: Estratificación y Repechaje

Para superar las limitaciones de los modelos pequeños (Quantized Small Models) y evitar "Cámaras de Eco", implementamos una lógica de selección avanzada:

1.  **Muestreo Estratificado:**
    Las soluciones no se mezclan al azar. Se agrupan según su afinidad vectorial con el "Experto" que las va a procesar. Esto simula la especialización profesional.

2.  **Mecanismo de Repechaje (The Rescue):**
    * **Problema:** La estratificación pura mata la creatividad (solo escuchas lo que quieres oír).
    * **Solución:** Utilizamos **Geometric Deep Learning (Curvatura de Ricci)** para identificar "Outliers Saludables".
    * **Lógica:** Si un nodo está lejos semánticamente (es raro) pero tiene buena estructura topológica (es coherente), el sistema lo **fuerza** a entrar en la siguiente ronda de agregación.
    * **Resultado:** Reintroducción controlada de varianza (creatividad) sin perder coherencia.

## 4. Stack Tecnológico

* **Hardware:** Optimizado para GPU única (RTX 4060).
* **Modelos Base:** LLMs pequeños cuantizados (Qwen 2.5/3 1.5B - 4B) actuando como "neuronas" o "expertos".
* **Librerías Críticas:** `GraphRicciCurvature` (Topología), `Sentence-Transformers` (Semántica), `NetworkX` (Grafo).

## 5. Hoja de Ruta Inmediata

1.  **Fase A (Gemini):** Construcción del "Benchmarking Harness" independiente. El sistema debe ser capaz de reportar si el RAS ahorró recursos en tareas fáciles (Semáforo Verde).
2.  **Fase B (Claude):** Implementación del `RSASolver` con la lógica de `stratified_sample` y `rescue_outliers`.
3.  **Fase C (Integración):** Conexión del Bandido (RAS) para controlar los knobs del RSA.

## 6. Conclusión de Investigación
Optamos por este modelo de agente porque permite que **modelos locales pequeños superen a modelos gigantes cerrados** mediante el tiempo de cómputo (Inference-time scaling). Al añadir el **Repechaje**, mitigamos el riesgo de alucinación colectiva, anclando la creatividad a la estabilidad topológica del grafo.
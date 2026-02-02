"""
RSA Solver - Motor principal de Recursive Self-Aggregation.

Implementa el algoritmo RSA de Venkatraman et al. con modificaciones
propietarias Antigravity: Estratificación semántica y Repechaje.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- population_size: N - tamaño de población inicial
- aggregation_size: K - candidatos por agregación
- steps: T - rondas de agregación
- temperature: Controla softmax de selección
- enable_repechage: Activa/desactiva mecanismo de rescate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Literal
import numpy as np

from src.rsa_engine.population import PopulationManager, Candidate
from src.rsa_engine.embeddings import EmbeddingProvider, MockEmbeddingProvider
from src.rsa_engine.selection import stratified_sample, RepechageBuffer
from src.core.backend import (
    SingleModelGenerator,
    MockSingleModelGenerator,
    EXPERT_ROLES,
    LLMBackend,
)
from src.verification.rsi_logger import RSILogger


# ============== LLM Adapter Interface ==============

class LLMAdapter(Protocol):
    """
    Protocolo para adaptadores de LLM.
    
    Implementa esta interfaz para conectar tu backend LLM
    (Qwen, vLLM, Ollama, etc.)
    """
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Genera una respuesta para el prompt."""
        ...
    
    def aggregate(self, query: str, responses: list[str], **kwargs) -> str:
        """
        Agrega múltiples respuestas en una síntesis.
        
        Args:
            query: Query original
            responses: Lista de respuestas a agregar
        
        Returns:
            Respuesta agregada/sintetizada
        """
        ...


# MockLLMAdapter movido a src/core/backend.py como MockSingleModelGenerator
# Mantener alias para compatibilidad
MockLLMAdapter = MockSingleModelGenerator


# ============== Configuration ==============

@dataclass
class RSAConfig:
    """
    Configuración del RSA Solver.
    
    Hiperparámetros principales que el RAS Controller puede ajustar.
    """
    population_size: int = 16  # N: tamaño de población inicial
    aggregation_size: int = 4  # K: candidatos por agregación
    steps: int = 3  # T: rondas de agregación
    temperature: float = 1.0  # PARAM: softmax temperature
    enable_repechage: bool = True  # PARAM: activar repechaje
    repechage_max_size: int = 5  # PARAM: buffer de repechaje
    repechage_distance_threshold: float = 0.7  # PARAM
    repechage_curvature_threshold: float = -2.0  # PARAM
    n_experts_per_round: int = 4  # Expertos activos por ronda
    seed: int | None = None


@dataclass
class RSAResult:
    """
    Resultado de una ejecución RSA.
    
    Contiene el mejor candidato final, métricas de proceso,
    y trazabilidad completa para análisis.
    """
    best_candidate: Candidate
    final_response: str
    total_candidates_generated: int
    total_aggregations: int
    rounds_completed: int
    repechage_stats: dict
    population_snapshot: PopulationManager
    metrics: dict = field(default_factory=dict)
    
    @property
    def cost_estimate(self) -> int:
        """Estimación de costo (N * T aproximado)."""
        return self.total_candidates_generated + self.total_aggregations


# ============== RSA Solver ==============

class RSASolver:
    """
    Motor RSA con Estratificación Semántica y Repechaje.
    
    Flujo de ejecución:
    1. Generar población inicial (N candidatos) para el query
    2. Para cada ronda t en T:
       a. Seleccionar "expertos" activos de la población
       b. Para cada experto:
          - stratified_sample(K candidatos) basado en similitud
          - Agregar respuestas → nuevo candidato
       c. Detectar outliers de alta calidad → RepechageBuffer
       d. Inyectar candidatos rescatados en siguiente ronda
    3. Retornar mejor candidato final
    
    La diferencia clave vs RSA original:
    - stratified_sample() en lugar de random.sample()
    - RepechageBuffer para evitar cámaras de eco
    """
    
    def __init__(
        self,
        llm_adapter: LLMAdapter | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: RSAConfig | None = None,
        logger: RSILogger | None = None,
    ):
        self.llm = llm_adapter or MockLLMAdapter()
        self.config = config or RSAConfig()
        self._embedding_provider = embedding_provider
        self._logger = logger or RSILogger("rsa_solver", console_output=False)
        self._rng = np.random.default_rng(self.config.seed)
    
    def solve(self, query: str) -> RSAResult:
        """
        Ejecuta RSA completo para un query.
        
        Args:
            query: Pregunta/tarea a resolver
        
        Returns:
            RSAResult con mejor respuesta y métricas
        """
        self._logger.log_simbolico(
            "rsa_solve_started",
            details={
                "query_length": len(query),
                "config": {
                    "N": self.config.population_size,
                    "K": self.config.aggregation_size,
                    "T": self.config.steps,
                },
            },
        )
        
        # Inicializar población
        population = PopulationManager(
            embedding_provider=self._embedding_provider,
            seed=self.config.seed,
            logger=self._logger,
        )
        
        # Inicializar buffer de repechaje
        repechage_buffer = RepechageBuffer(
            max_size=self.config.repechage_max_size,
            distance_threshold=self.config.repechage_distance_threshold,
            curvature_stability_threshold=self.config.repechage_curvature_threshold,
        )
        
        # === Paso 1: Generar población inicial ===
        self._generate_initial_population(query, population)
        
        # === Paso 2: Rondas de agregación ===
        total_aggregations = 0
        
        for round_num in range(1, self.config.steps + 1):
            aggregations_this_round = self._run_aggregation_round(
                query=query,
                population=population,
                repechage_buffer=repechage_buffer,
                round_num=round_num,
            )
            total_aggregations += aggregations_this_round
            
            self._logger.log_simbolico(
                f"rsa_round_{round_num}_completed",
                metrics={
                    "aggregations": aggregations_this_round,
                    "population_size": population.size,
                    "rescued_this_round": len(repechage_buffer.rescued_candidates),
                },
            )
        
        # === Paso 3: Seleccionar mejor candidato ===
        # Actualizar curvaturas finales
        population.update_all_curvatures()
        
        # Mejor por score (podrías usar otra métrica)
        best_candidates = population.get_top_candidates(1, by="score")
        
        if best_candidates:
            best = best_candidates[0]
        else:
            # Fallback: último creado
            best = list(population.candidates.values())[-1]
        
        result = RSAResult(
            best_candidate=best,
            final_response=best.response,
            total_candidates_generated=population.size,
            total_aggregations=total_aggregations,
            rounds_completed=self.config.steps,
            repechage_stats=repechage_buffer.get_rescue_stats(),
            population_snapshot=population,
            metrics={
                "config": {
                    "N": self.config.population_size,
                    "K": self.config.aggregation_size,
                    "T": self.config.steps,
                },
                "enable_repechage": self.config.enable_repechage,
            },
        )
        
        self._logger.log_simbolico(
            "rsa_solve_completed",
            metrics={
                "total_candidates": population.size,
                "total_aggregations": total_aggregations,
                "cost_estimate": result.cost_estimate,
            },
        )
        
        return result
    
    def _generate_initial_population(
        self,
        query: str,
        population: PopulationManager,
    ) -> None:
        """
        Genera la población inicial de N candidatos.
        
        Simula "expertos" rotando system_role dinámicamente
        para obtener diversidad con un solo modelo.
        """
        # Roles disponibles para rotación
        available_roles = list(EXPERT_ROLES.keys())
        n_roles = len(available_roles)
        
        for i in range(self.config.population_size):
            # Rotar role para diversidad
            role_idx = i % n_roles
            system_role = available_roles[role_idx]
            
            # Generar respuesta con rol específico y alta temperatura
            prompt = self._format_generation_prompt(query, i)
            response = self.llm.generate(
                prompt=prompt,
                system_role=system_role,
                temperature=0.8,  # Alta para diversidad en población inicial
            )
            
            # Añadir a población
            cid = population.add_candidate(
                response=response,
                round_created=0,
                metadata={
                    "generation_index": i,
                    "expert_role": system_role,
                },
            )
            
            # Score inicial (placeholder - podría venir del LLM)
            population.candidates[cid].score = self._rng.random()
        
        # Calcular curvaturas iniciales
        population.update_all_curvatures()
    
    def _run_aggregation_round(
        self,
        query: str,
        population: PopulationManager,
        repechage_buffer: RepechageBuffer,
        round_num: int,
    ) -> int:
        """
        Ejecuta una ronda de agregación.
        
        Returns:
            Número de agregaciones realizadas
        """
        # Obtener candidatos de la ronda anterior
        prev_round = round_num - 1
        candidate_pool = [
            cid for cid, c in population.candidates.items()
            if c.round_created == prev_round
        ]
        
        # Inyectar candidatos rescatados
        if self.config.enable_repechage:
            candidate_pool = repechage_buffer.inject_rescued(candidate_pool)
        
        if len(candidate_pool) < self.config.aggregation_size:
            # No hay suficientes candidatos
            return 0
        
        # Seleccionar expertos para esta ronda
        n_experts = min(self.config.n_experts_per_round, len(candidate_pool))
        expert_ids = list(self._rng.choice(
            candidate_pool,
            size=n_experts,
            replace=False,
        ))
        
        aggregation_count = 0
        all_selected_this_round = []
        
        for expert_id in expert_ids:
            # Obtener perfil del experto
            expert_embedding = population.get_expert_profile_embedding(expert_id)
            
            # Pool excluyendo al experto
            pool_for_expert = [cid for cid in candidate_pool if cid != expert_id]
            
            if len(pool_for_expert) < self.config.aggregation_size:
                continue
            
            # Selección estratificada
            selected = stratified_sample(
                population=population,
                expert_embedding=expert_embedding,
                k=self.config.aggregation_size,
                candidate_pool=pool_for_expert,
                temperature=self.config.temperature,
                seed=self._rng.integers(0, 2**31),
                logger=self._logger,
            )
            
            all_selected_this_round.extend(selected)
            
            # Agregar respuestas
            responses_to_aggregate = [
                population.candidates[cid].response for cid in selected
            ]
            
            aggregated_response = self.llm.aggregate(
                query=query,
                responses=responses_to_aggregate,
            )
            
            # Crear nuevo candidato
            new_cid = population.add_candidate(
                response=aggregated_response,
                round_created=round_num,
                parent_ids=selected,
                score=self._estimate_aggregation_score(selected, population),
            )
            
            aggregation_count += 1
        
        # Detectar outliers para repechaje
        if self.config.enable_repechage:
            repechage_buffer.detect_and_rescue(
                population=population,
                current_aggregation=all_selected_this_round,
                excluded_ids=set(expert_ids),
            )
        
        return aggregation_count
    
    def _format_generation_prompt(self, query: str, index: int) -> str:
        """Formatea prompt para generación inicial."""
        return f"[Generation {index}] Please provide a thoughtful response to: {query}"
    
    def _estimate_aggregation_score(
        self,
        parent_ids: list[int],
        population: PopulationManager,
    ) -> float:
        """
        Estima score de un candidato agregado basado en sus padres.
        
        Heurística: promedio de scores de padres + bonus por diversidad.
        """
        if not parent_ids:
            return 0.5
        
        parent_scores = [
            population.candidates[pid].score
            for pid in parent_ids
            if pid in population.candidates
        ]
        
        if not parent_scores:
            return 0.5
        
        avg_score = np.mean(parent_scores)
        
        # Bonus por diversidad (varianza de curvaturas de padres)
        parent_curvatures = [
            population.candidates[pid].ricci_curvature
            for pid in parent_ids
            if pid in population.candidates
        ]
        
        if len(parent_curvatures) > 1:
            diversity_bonus = min(0.1, np.std(parent_curvatures) * 0.05)
        else:
            diversity_bonus = 0.0
        
        return float(avg_score + diversity_bonus)

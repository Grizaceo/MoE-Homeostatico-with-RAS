"""
L_kn Manager - El Gerente Homeostático.

Implementa el Sistema de Activación Reticular (RAS) para controlar
el presupuesto computacional del RSA Engine.

Objetivo: Minimizar (N * T) mientras se mantiene la confianza simbólica.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- budget_profiles: Perfiles predefinidos de presupuesto
- complexity_thresholds: Umbrales para clasificar complejidad de queries
- adaptation_rate: Velocidad de adaptación del modelo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol
import numpy as np

from src.verification.rsi_logger import RSILogger


# ============== Data Classes ==============

class BudgetProfile(str, Enum):
    """Perfiles de presupuesto predefinidos."""
    MINIMAL = "minimal"
    LIGHT = "light"
    STANDARD = "standard"
    INTENSIVE = "intensive"


@dataclass
class RSABudget:
    """
    Presupuesto decidido por RAS para una ejecución RSA.
    
    Attributes:
        population_size: N - tamaño de población inicial
        aggregation_size: K - candidatos por agregación
        steps: T - rondas de agregación
        profile: Perfil de presupuesto usado
        confidence: Confianza del RAS en esta decisión
    """
    population_size: int  # N
    aggregation_size: int  # K
    steps: int  # T
    profile: BudgetProfile = BudgetProfile.STANDARD
    confidence: float = 0.8
    
    @property
    def total_cost(self) -> int:
        """
        Estimación de costo computacional.
        
        Fórmula: N + (N * T) aproximadamente
        (generaciones iniciales + agregaciones por ronda)
        """
        return self.population_size * self.steps
    
    @property
    def efficiency_ratio(self) -> float:
        """Ratio K/N - qué fracción de población se usa."""
        if self.population_size == 0:
            return 0.0
        return self.aggregation_size / self.population_size


@dataclass
class QueryContext:
    """
    Contexto del query para decisión de presupuesto.
    
    El RAS analiza este contexto para decidir cuánto invertir.
    """
    query: str
    query_embedding: np.ndarray | None = None
    estimated_complexity: float = 0.5  # 0=trivial, 1=muy complejo
    confidence_threshold: float = 0.8  # Confianza mínima requerida
    max_cost: int | None = None  # Límite de costo (opcional)
    metadata: dict = field(default_factory=dict)


@dataclass
class OutcomeRecord:
    """Registro de un resultado para aprendizaje."""
    context: QueryContext
    budget_used: RSABudget
    final_confidence: float
    success: bool
    actual_cost: int


# ============== RAS Controller ==============

class L_kn_Manager:
    """
    Controlador homeostático de recursos.

    Su nombre 'L-kn' (Lacanian Knot) referencia la función topológica
    de estabilización del sistema (RSI - Real, Simbólico, Imaginario).

    Rol: Homeostasis y Economía de la Atención.
    No resuelve el problema, solo asigna presupuesto.

    Decide dinámicamente:
        N: Population Size (Amplitud de ideas)
        K: Aggregation Size (Presión de convergencia)
        T: Iteration Steps (Profundidad de refinamiento)
        
    Basado en:
        1. Energía disponible (ComputeConstraints - VRAM/Time)
        2. Complejidad percibida del input
        3. Historial de éxito/fallo (Feedback Loop)
    """
    
    # Perfiles predefinidos de presupuesto
    BUDGET_PROFILES: dict[BudgetProfile, RSABudget] = {
        BudgetProfile.MINIMAL: RSABudget(
            population_size=4, aggregation_size=2, steps=1,
            profile=BudgetProfile.MINIMAL
        ),
        BudgetProfile.LIGHT: RSABudget(
            population_size=8, aggregation_size=2, steps=2,
            profile=BudgetProfile.LIGHT
        ),
        BudgetProfile.STANDARD: RSABudget(
            population_size=16, aggregation_size=4, steps=3,
            profile=BudgetProfile.STANDARD
        ),
        BudgetProfile.INTENSIVE: RSABudget(
            population_size=32, aggregation_size=8, steps=5,
            profile=BudgetProfile.INTENSIVE
        ),
    }
    
    # Umbrales de complejidad para selección de perfil
    COMPLEXITY_THRESHOLDS: dict[BudgetProfile, float] = {
        BudgetProfile.MINIMAL: 0.25,  # complexity < 0.25
        BudgetProfile.LIGHT: 0.5,     # 0.25 <= complexity < 0.5
        BudgetProfile.STANDARD: 0.75, # 0.5 <= complexity < 0.75
        BudgetProfile.INTENSIVE: 1.0, # complexity >= 0.75
    }
    
    def __init__(
        self,
        default_profile: BudgetProfile = BudgetProfile.STANDARD,
        enable_adaptation: bool = True,  # PARAM: activar aprendizaje
        adaptation_rate: float = 0.1,  # PARAM: velocidad de adaptación
        seed: int | None = None,
        logger: RSILogger | None = None,
    ):
        self.default_profile = default_profile
        self.enable_adaptation = enable_adaptation
        self.adaptation_rate = adaptation_rate
        self._rng = np.random.default_rng(seed)
        self._logger = logger or RSILogger("ras_controller", console_output=False)
        
        # Historial para aprendizaje
        self._outcome_history: list[OutcomeRecord] = []
        
        # Ajustes aprendidos (futuro: Thompson Sampling priors)
        self._profile_success_rates: dict[BudgetProfile, float] = {
            p: 0.8 for p in BudgetProfile
        }
    
    def decide_budget(self, context: QueryContext) -> RSABudget:
        """
        Decide el presupuesto basado en el contexto del query.
        
        Heurística actual:
        1. Estimar complejidad si no está dada
        2. Mapear complejidad a perfil de presupuesto
        3. Ajustar por límite de costo si aplica
        4. Añadir confianza basada en historial
        
        Args:
            context: Contexto del query
        
        Returns:
            RSABudget con N, K, T decididos
        """
        # Estimar complejidad si no está dada
        complexity = context.estimated_complexity
        if complexity == 0.5 and context.query_embedding is None:
            complexity = self._estimate_complexity_from_query(context.query)
        
        # Mapear a perfil
        profile = self._select_profile(complexity)
        
        # Obtener budget base
        budget = self._get_budget_for_profile(profile)
        
        # Ajustar por límite de costo
        if context.max_cost is not None:
            budget = self._constrain_to_max_cost(budget, context.max_cost)
        
        # Calcular confianza
        confidence = self._compute_confidence(profile, complexity)
        budget.confidence = confidence
        
        self._logger.log_simbolico(
            "budget_decided",
            details={
                "query_length": len(context.query),
                "estimated_complexity": complexity,
                "selected_profile": profile.value,
            },
            metrics={
                "N": budget.population_size,
                "K": budget.aggregation_size,
                "T": budget.steps,
                "total_cost": budget.total_cost,
                "confidence": confidence,
            },
        )
        
        return budget
    
    def _estimate_complexity_from_query(self, query: str) -> float:
        """
        Estima complejidad del query basado en heurísticas simples.
        
        Señales usadas:
        - Longitud del query
        - Presencia de palabras clave complejas
        - Signos de interrogación múltiples
        """
        # Normalizar longitud (0 para corto, 1 para muy largo)
        length_score = min(1.0, len(query) / 500)
        
        # Palabras clave que indican complejidad
        complex_keywords = [
            "compare", "analyze", "explain", "evaluate",
            "compara", "analiza", "explica", "evalúa",
            "why", "how", "por qué", "cómo",
            "pros and cons", "trade-off", "implications",
        ]
        
        query_lower = query.lower()
        keyword_count = sum(1 for kw in complex_keywords if kw in query_lower)
        keyword_score = min(1.0, keyword_count / 3)
        
        # Múltiples preguntas
        question_marks = query.count("?")
        multi_question_score = min(1.0, question_marks / 3)
        
        # Combinar (pesos ajustables)
        complexity = (
            0.3 * length_score +
            0.5 * keyword_score +
            0.2 * multi_question_score
        )
        
        return float(complexity)
    
    def _select_profile(self, complexity: float) -> BudgetProfile:
        """Mapea complejidad a perfil de presupuesto."""
        if complexity < self.COMPLEXITY_THRESHOLDS[BudgetProfile.MINIMAL]:
            return BudgetProfile.MINIMAL
        elif complexity < self.COMPLEXITY_THRESHOLDS[BudgetProfile.LIGHT]:
            return BudgetProfile.LIGHT
        elif complexity < self.COMPLEXITY_THRESHOLDS[BudgetProfile.STANDARD]:
            return BudgetProfile.STANDARD
        else:
            return BudgetProfile.INTENSIVE
    
    def _get_budget_for_profile(self, profile: BudgetProfile) -> RSABudget:
        """Obtiene budget para un perfil (copia para evitar mutación)."""
        base = self.BUDGET_PROFILES[profile]
        return RSABudget(
            population_size=base.population_size,
            aggregation_size=base.aggregation_size,
            steps=base.steps,
            profile=profile,
            confidence=base.confidence,
        )
    
    def _constrain_to_max_cost(
        self,
        budget: RSABudget,
        max_cost: int,
    ) -> RSABudget:
        """Ajusta budget para respetar límite de costo."""
        if budget.total_cost <= max_cost:
            return budget
        
        # Reducir T primero, luego N
        new_steps = budget.steps
        new_pop = budget.population_size
        
        while new_pop * new_steps > max_cost and new_steps > 1:
            new_steps -= 1
        
        while new_pop * new_steps > max_cost and new_pop > 2:
            new_pop = new_pop // 2
        
        budget.population_size = new_pop
        budget.steps = new_steps
        budget.aggregation_size = min(budget.aggregation_size, new_pop // 2)
        
        return budget
    
    def _compute_confidence(
        self,
        profile: BudgetProfile,
        complexity: float,
    ) -> float:
        """Computa confianza en la decisión."""
        # Base: tasa de éxito histórica del perfil
        base_confidence = self._profile_success_rates[profile]
        
        # Penalizar si la complejidad está en el borde del umbral
        threshold = self.COMPLEXITY_THRESHOLDS[profile]
        distance_to_threshold = abs(complexity - threshold)
        edge_penalty = max(0, 0.1 - distance_to_threshold)
        
        return float(base_confidence - edge_penalty)
    
    def update_from_outcome(
        self,
        context: QueryContext,
        budget_used: RSABudget,
        final_confidence: float,
        success: bool,
    ) -> None:
        """
        Actualiza el modelo basado en el resultado de una ejecución.
        
        Usado para aprendizaje online (futuro: Contextual Bandit).
        
        Args:
            context: Contexto del query original
            budget_used: Budget que se usó
            final_confidence: Confianza final del RSA
            success: Si se consideró exitoso
        """
        if not self.enable_adaptation:
            return
        
        # Registrar outcome
        record = OutcomeRecord(
            context=context,
            budget_used=budget_used,
            final_confidence=final_confidence,
            success=success,
            actual_cost=budget_used.total_cost,
        )
        self._outcome_history.append(record)
        
        # Actualizar tasa de éxito del perfil (exponential moving average)
        profile = budget_used.profile
        old_rate = self._profile_success_rates[profile]
        new_rate = (
            (1 - self.adaptation_rate) * old_rate +
            self.adaptation_rate * (1.0 if success else 0.0)
        )
        self._profile_success_rates[profile] = new_rate
        
        self._logger.log_simbolico(
            "outcome_recorded",
            details={
                "profile": profile.value,
                "success": success,
                "final_confidence": final_confidence,
            },
            metrics={
                "new_success_rate": new_rate,
                "history_size": len(self._outcome_history),
            },
        )
    
    def get_stats(self) -> dict:
        """Retorna estadísticas del controlador."""
        if not self._outcome_history:
            return {
                "total_decisions": 0,
                "success_rate": 0.0,
                "profile_distribution": {},
            }
        
        total = len(self._outcome_history)
        successes = sum(1 for r in self._outcome_history if r.success)
        
        profile_counts = {}
        for record in self._outcome_history:
            p = record.budget_used.profile.value
            profile_counts[p] = profile_counts.get(p, 0) + 1
        
        return {
            "total_decisions": total,
            "success_rate": successes / total,
            "avg_cost": np.mean([r.actual_cost for r in self._outcome_history]),
            "profile_distribution": profile_counts,
            "learned_success_rates": {
                p.value: rate for p, rate in self._profile_success_rates.items()
            },
        }
    
    def reset_learning(self) -> None:
        """Resetea el historial y tasas aprendidas."""
        self._outcome_history.clear()
        self._profile_success_rates = {p: 0.8 for p in BudgetProfile}


# Alias de compatibilidad hacia atrás
RASController = L_kn_Manager

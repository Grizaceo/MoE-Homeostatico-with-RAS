"""
Lógica de Selección para RSA: Estratificación y Repechaje.

Este es el CORE de la modificación propietaria Antigravity:
- stratified_sample(): Selección por similitud semántica (no random.sample)
- RepechageBuffer: Rescate de outliers de alta calidad

VARIABLES AJUSTABLES (marcadas con # PARAM):
- temperature: Controla "dureza" de la distribución de selección
- distance_threshold: Umbral para considerar un candidato como outlier
- curvature_stability_threshold: Curvatura mínima para outlier "saludable"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

from src.verification.rsi_logger import RSILogger

if TYPE_CHECKING:
    from src.rsa_engine.population import PopulationManager, Candidate


# ============== Stratified Sampling ==============

def compute_similarity_scores(
    population: "PopulationManager",
    expert_embedding: np.ndarray,
    candidate_ids: list[int],
) -> dict[int, float]:
    """
    Calcula similitud coseno de cada candidato con el perfil del experto.
    
    Args:
        population: Manager de población
        expert_embedding: Embedding del experto agregador
        candidate_ids: IDs de candidatos a evaluar
    
    Returns:
        Diccionario id -> similitud coseno
    """
    scores = {}
    for cid in candidate_ids:
        if cid not in population.candidates:
            continue
        candidate = population.candidates[cid]
        # Similitud coseno (embeddings ya normalizados)
        similarity = float(np.dot(candidate.embedding, expert_embedding))
        scores[cid] = similarity
    return scores


def stratified_sample(
    population: "PopulationManager",
    expert_embedding: np.ndarray,
    k: int,
    candidate_pool: list[int] | None = None,
    temperature: float = 1.0,  # PARAM: controla concentración
    seed: int | None = None,
    logger: RSILogger | None = None,
) -> list[int]:
    """
    Selección estratificada por similitud semántica.
    
    En lugar de random.sample(), selecciona candidatos con probabilidad
    proporcional a su similitud coseno con el perfil del experto.
    
    Distribución: softmax(similarity / temperature)
    - temperature baja → selección más determinista (favorece muy similares)
    - temperature alta → selección más uniforme
    
    Esta función simula "especialización profesional": expertos tienden
    a agregar respuestas que ya entienden (cercanas semánticamente).
    
    Args:
        population: Manager de población
        expert_embedding: Embedding del experto agregador
        k: Número de candidatos a seleccionar
        candidate_pool: Pool de IDs elegibles (None = todos)
        temperature: Parámetro de softmax
        seed: Semilla para reproducibilidad
        logger: Logger RSI
    
    Returns:
        Lista de k IDs seleccionados
    """
    if logger is None:
        logger = RSILogger("selection", console_output=False)
    
    rng = np.random.default_rng(seed)
    
    # Pool de candidatos
    if candidate_pool is None:
        candidate_pool = list(population.candidates.keys())
    
    if len(candidate_pool) <= k:
        # No hay suficientes candidatos, retornar todos
        return candidate_pool.copy()
    
    # Calcular similitudes
    similarities = compute_similarity_scores(
        population, expert_embedding, candidate_pool
    )
    
    if not similarities:
        return []
    
    # Convertir a arrays para softmax
    ids = list(similarities.keys())
    sims = np.array([similarities[i] for i in ids])
    
    # Shift para estabilidad numérica (evitar overflow en exp)
    sims_shifted = (sims - np.max(sims)) / temperature
    
    # Softmax
    exp_sims = np.exp(sims_shifted)
    probs = exp_sims / np.sum(exp_sims)
    
    # Muestreo sin reemplazo
    selected_indices = rng.choice(
        len(ids),
        size=min(k, len(ids)),
        replace=False,
        p=probs,
    )
    
    selected = [ids[i] for i in selected_indices]
    
    # Log de estadísticas
    logger.log_simbolico(
        "stratified_sample_completed",
        details={
            "pool_size": len(candidate_pool),
            "k_requested": k,
            "k_selected": len(selected),
            "temperature": temperature,
        },
        metrics={
            "mean_similarity": float(np.mean(sims)),
            "max_similarity": float(np.max(sims)),
            "min_similarity": float(np.min(sims)),
            "entropy": float(-np.sum(probs * np.log(probs + 1e-10))),
        },
    )
    
    return selected


# ============== Repechaje (The Rescue) ==============

@dataclass
class RepechageBuffer:
    """
    Buffer de "Repechaje" para outliers de alta calidad.
    
    PROBLEMA: La estratificación pura mata la creatividad porque solo
    selecciona respuestas similares al experto (cámara de eco).
    
    SOLUCIÓN: Detectar candidatos que son:
    1. Semánticamente distantes del consenso actual (outliers)
    2. Pero tienen curvatura Ricci estable (no son ruido)
    
    Estos candidatos se inyectan forzosamente en la siguiente ronda
    para evitar convergencia prematura a soluciones mediocres.
    
    Lógica: "Si es diferente pero coherente, merece una segunda oportunidad."
    
    Attributes:
        max_size: Tamaño máximo del buffer
        distance_threshold: Distancia semántica mínima para ser outlier
        curvature_stability_threshold: Curvatura mínima para ser "saludable"
        rescued_candidates: IDs de candidatos rescatados
    """
    
    max_size: int = 5  # PARAM: tamaño máximo del buffer
    distance_threshold: float = 0.7  # PARAM: distancia para ser outlier
    curvature_stability_threshold: float = -2.0  # PARAM: curvatura mínima
    
    rescued_candidates: list[int] = field(default_factory=list)
    rescue_history: list[dict] = field(default_factory=list)
    
    _logger: RSILogger = field(
        default_factory=lambda: RSILogger("repechage", console_output=False)
    )
    
    def detect_and_rescue(
        self,
        population: "PopulationManager",
        current_aggregation: list[int],
        excluded_ids: set[int] | None = None,
    ) -> list[int]:
        """
        Detecta outliers de alta calidad y los guarda para repechaje.
        
        Un candidato es "rescatable" si:
        1. NO fue seleccionado en la agregación actual
        2. Está semánticamente lejos del centroide actual
        3. Tiene curvatura Ricci >= threshold (topológicamente estable)
        
        Args:
            population: Manager de población
            current_aggregation: IDs seleccionados en ronda actual
            excluded_ids: IDs a excluir de consideración
        
        Returns:
            IDs de candidatos rescatados en esta ronda
        """
        if excluded_ids is None:
            excluded_ids = set()
        
        # Calcular centroide de la agregación actual
        if current_aggregation:
            agg_embeddings = np.array([
                population.candidates[cid].embedding 
                for cid in current_aggregation
                if cid in population.candidates
            ])
            if len(agg_embeddings) > 0:
                centroid = np.mean(agg_embeddings, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
            else:
                centroid = population.get_centroid_embedding()
        else:
            centroid = population.get_centroid_embedding()
        
        # Buscar outliers de alta calidad
        new_rescued = []
        current_set = set(current_aggregation)
        already_rescued = set(self.rescued_candidates)
        
        for cid, candidate in population.candidates.items():
            # Excluir ya seleccionados, ya rescatados, y excluidos
            if cid in current_set or cid in already_rescued or cid in excluded_ids:
                continue
            
            # Criterio 1: Distancia semántica al centroide
            distance = 1.0 - float(np.dot(candidate.embedding, centroid))
            
            if distance < self.distance_threshold:
                continue  # No es outlier
            
            # Criterio 2: Curvatura Ricci estable
            # Asegurar que la curvatura está calculada
            if candidate.ricci_curvature == 0.0:
                population.compute_local_curvature(cid)
            
            curvature = candidate.ricci_curvature
            
            if curvature < self.curvature_stability_threshold:
                continue  # Muy inestable topológicamente (ruido)
            
            # ¡Candidato rescatable!
            new_rescued.append(cid)
            
            self.rescue_history.append({
                "candidate_id": cid,
                "distance_to_centroid": distance,
                "curvature": curvature,
                "round": len(self.rescue_history),
            })
        
        # Limitar tamaño del buffer
        space_available = self.max_size - len(self.rescued_candidates)
        new_rescued = new_rescued[:space_available]
        
        self.rescued_candidates.extend(new_rescued)
        
        # Log
        self._logger.log_simbolico(
            "rescue_detection_completed",
            details={
                "candidates_evaluated": len(population.candidates) - len(current_set),
                "rescued_this_round": len(new_rescued),
                "total_in_buffer": len(self.rescued_candidates),
                "distance_threshold": self.distance_threshold,
                "curvature_threshold": self.curvature_stability_threshold,
            },
        )
        
        return new_rescued
    
    def inject_rescued(
        self,
        next_round_candidates: list[int],
        max_inject: int | None = None,
    ) -> list[int]:
        """
        Inyecta candidatos rescatados en la siguiente ronda.
        
        Args:
            next_round_candidates: Lista actual de candidatos para próxima ronda
            max_inject: Máximo a inyectar (None = todos los rescatados)
        
        Returns:
            Nueva lista con candidatos inyectados
        """
        if not self.rescued_candidates:
            return next_round_candidates
        
        to_inject = self.rescued_candidates[:max_inject] if max_inject else self.rescued_candidates
        
        # Evitar duplicados
        existing = set(next_round_candidates)
        injected = [cid for cid in to_inject if cid not in existing]
        
        # Limpiar buffer de los inyectados
        injected_set = set(injected)
        self.rescued_candidates = [
            cid for cid in self.rescued_candidates if cid not in injected_set
        ]
        
        result = next_round_candidates + injected
        
        self._logger.log_simbolico(
            "rescue_injection_completed",
            details={
                "injected_count": len(injected),
                "remaining_in_buffer": len(self.rescued_candidates),
            },
        )
        
        return result
    
    def clear(self) -> None:
        """Limpia el buffer de rescatados."""
        self.rescued_candidates.clear()
    
    def get_rescue_stats(self) -> dict:
        """Retorna estadísticas del repechaje."""
        if not self.rescue_history:
            return {
                "total_rescued": 0,
                "avg_distance": 0.0,
                "avg_curvature": 0.0,
            }
        
        distances = [h["distance_to_centroid"] for h in self.rescue_history]
        curvatures = [h["curvature"] for h in self.rescue_history]
        
        return {
            "total_rescued": len(self.rescue_history),
            "avg_distance": float(np.mean(distances)),
            "avg_curvature": float(np.mean(curvatures)),
            "max_distance": float(np.max(distances)),
            "min_curvature": float(np.min(curvatures)),
        }


# ============== Helpers ==============

def random_sample_baseline(
    population: "PopulationManager",
    k: int,
    candidate_pool: list[int] | None = None,
    seed: int | None = None,
) -> list[int]:
    """
    Muestreo aleatorio baseline (para comparación con stratified).
    
    Esta es la versión "vanilla" del paper RSA original.
    """
    rng = np.random.default_rng(seed)
    
    if candidate_pool is None:
        candidate_pool = list(population.candidates.keys())
    
    if len(candidate_pool) <= k:
        return candidate_pool.copy()
    
    return list(rng.choice(candidate_pool, size=k, replace=False))

"""
Mixture of Experts (MoE) Gating para espacios geométricos.

Este módulo implementa un sistema de gating que selecciona dinámicamente
qué espacio geométrico usar para cada nodo/arista basándose en
características locales (curvatura, grado, etc.).

El RAS (Real-Simbólico-Imaginario) actúa como meta-control.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- gating_temperature: Suavidad del softmax
- expert_weights: Pesos iniciales de expertos
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Callable
from enum import Enum

from src.geometry.euclidean import embed_euclidean, compute_embedding_distance, EmbeddingResult
from src.geometry.hyperbolic import embed_hyperbolic, compute_hyperbolic_distance, HyperbolicEmbeddingResult
from src.core.curvature import compute_forman_ricci
from src.verification.rsi_logger import RSILogger


class GeometrySpace(Enum):
    """Espacios geométricos disponibles."""
    EUCLIDEAN = "E"
    HYPERBOLIC = "H"
    # SPHERICAL = "S"  # Futuro


@dataclass
class NodeFeatures:
    """Características de un nodo para decisión de gating."""
    node: int
    degree: int
    local_curvature: float  # Curvatura promedio de aristas incidentes
    clustering: float
    is_hub: bool  # degree > median degree


@dataclass
class GatingDecision:
    """Resultado de una decisión de gating."""
    node: int
    weights: dict[str, float]  # {space: weight}
    selected: str  # Espacio seleccionado (argmax)
    confidence: float  # max weight
    features: NodeFeatures


@dataclass  
class GeometricMoEResult:
    """Resultado del embedding con MoE."""
    positions_euclidean: dict[int, np.ndarray]
    positions_hyperbolic: dict[int, np.ndarray]
    gating_decisions: dict[int, GatingDecision]
    mixed_positions: dict[int, np.ndarray]  # Combinación ponderada
    metrics: dict


def compute_node_features(
    G: nx.Graph,
    curvatures: dict[tuple, float] | None = None,
) -> dict[int, NodeFeatures]:
    """
    Calcula características de cada nodo para decisión de gating.
    
    Args:
        G: Grafo
        curvatures: Curvaturas pre-calculadas (opcional)
    
    Returns:
        Diccionario {node: NodeFeatures}
    """
    if curvatures is None:
        curvatures = compute_forman_ricci(G)
    
    # Estadísticas globales
    degrees = dict(G.degree())
    median_degree = np.median(list(degrees.values()))
    clustering = nx.clustering(G)
    
    features = {}
    
    for node in G.nodes():
        # Curvatura local: promedio de curvaturas de aristas incidentes
        incident_curvatures = [
            curvatures.get((node, neighbor), curvatures.get((neighbor, node), 0))
            for neighbor in G.neighbors(node)
        ]
        local_curv = np.mean(incident_curvatures) if incident_curvatures else 0
        
        features[node] = NodeFeatures(
            node=node,
            degree=degrees[node],
            local_curvature=local_curv,
            clustering=clustering[node],
            is_hub=degrees[node] > median_degree,
        )
    
    return features


def soft_gating(
    features: NodeFeatures,
    temperature: float = 1.0,  # PARAM: menor = más duro
    curvature_weight: float = 0.7,  # PARAM: peso de curvatura (AUMENTADO)
    clustering_weight: float = 0.25,  # PARAM: peso de clustering
    degree_weight: float = 0.05,  # PARAM: peso de grado (REDUCIDO)
) -> dict[str, float]:
    """
    Calcula pesos soft para cada espacio basado en características del nodo.
    
    Heurísticas corregidas (v2):
    - Curvatura negativa + bajo clustering → Hiperbólico (árbol/jerarquía)
    - Curvatura positiva o alta clustering → Euclídeo (comunidades densas)
    - Grado tiene peso menor (no discrimina bien entre espacios)
    
    La clave es que ÁRBOLES tienen:
    - Curvatura Forman negativa en nodos internos
    - Clustering = 0 (sin triángulos)
    
    Args:
        features: Características del nodo
        temperature: Temperatura del softmax
        curvature_weight, clustering_weight, degree_weight: Pesos de features
    
    Returns:
        Diccionario {space: weight} que suma 1
    """
    # ===== Señales para Hiperbólico =====
    # 1. Curvatura muy negativa → estructura jerárquica
    #    Forman: F = 4 - deg(u) - deg(v)
    #    En árboles: internos tienen F ≤ 0, hojas conectadas a internos tienen F = 1
    curv = features.local_curvature
    
    # Normalizar: curvaturas muy negativas (<-2) → H, positivas → E
    if curv < -2:
        curv_signal_H = 1.0  # Muy jerárquico
    elif curv < 0:
        curv_signal_H = 0.5 + (-curv) / 4  # Moderadamente jerárquico
    elif curv == 0:
        curv_signal_H = 0.5  # Neutral
    else:
        curv_signal_H = 0.3  # Poco jerárquico
    
    # 2. Clustering bajo → sin triángulos → estructura de árbol
    #    Árboles tienen clustering = 0
    clustering_signal_H = 1 - features.clustering  # 0 clustering → 1.0
    
    # 3. Grado (señal débil) - alto grado puede ser hub jerárquico
    #    Pero también puede ser nodo denso en comunidad
    degree_signal_H = min(features.degree / 20, 0.5)  # Capped a 0.5
    
    # ===== Score para Hiperbólico =====
    score_H = (
        curvature_weight * curv_signal_H +
        clustering_weight * clustering_signal_H +
        degree_weight * degree_signal_H
    )
    
    # ===== Score para Euclídeo =====
    score_E = (
        curvature_weight * (1 - curv_signal_H) +
        clustering_weight * features.clustering +
        degree_weight * (1 - degree_signal_H)
    )
    
    # Softmax
    scores = np.array([score_E, score_H])
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / exp_scores.sum()
    
    return {
        GeometrySpace.EUCLIDEAN.value: float(weights[0]),
        GeometrySpace.HYPERBOLIC.value: float(weights[1]),
    }


def geometric_moe_embedding(
    G: nx.Graph,
    dim: int = 32,  # PARAM
    temperature: float = 1.0,  # PARAM: gating temperature
    euclidean_method: str = "mds",  # PARAM
    hyperbolic_curvature: float = -1.0,  # PARAM
    hyperbolic_epochs: int = 50,  # PARAM: reducido para notebook
    seed: int | None = None,
    logger: RSILogger | None = None,
) -> GeometricMoEResult:
    """
    Embedding con Mixture of Experts geométrico.
    
    Combina embeddings Euclídeo e Hiperbólico con gating por nodo.
    
    Args:
        G: Grafo
        dim: Dimensionalidad de cada embedding
        temperature: Temperatura del gating
        euclidean_method: Método para embedding Euclídeo
        hyperbolic_curvature: Curvatura del espacio hiperbólico
        hyperbolic_epochs: Épocas de entrenamiento hiperbólico
        seed: Semilla
        logger: RSI logger
    
    Returns:
        GeometricMoEResult con embeddings y decisiones de gating
    """
    if logger is None:
        logger = RSILogger("moe_gating", console_output=False)
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    logger.log_simbolico(
        "moe_embedding_start",
        details={"n_nodes": n, "dim": dim, "temperature": temperature},
    )
    
    # 1. Calcular características para gating
    curvatures = compute_forman_ricci(G, logger=logger)
    node_features = compute_node_features(G, curvatures)
    
    # 2. Calcular embedding Euclídeo
    logger.log_simbolico("computing_euclidean", details={"method": euclidean_method})
    emb_euclidean = embed_euclidean(G, dim=dim, method=euclidean_method, logger=logger)
    
    # 3. Calcular embedding Hiperbólico
    logger.log_simbolico("computing_hyperbolic", details={"epochs": hyperbolic_epochs})
    emb_hyperbolic = embed_hyperbolic(
        G, dim=dim, 
        curvature=hyperbolic_curvature,
        epochs=hyperbolic_epochs,
        seed=seed,
        logger=logger,
    )
    
    # 4. Gating por nodo
    gating_decisions = {}
    mixed_positions = {}
    
    for node in nodes:
        features = node_features[node]
        weights = soft_gating(features, temperature=temperature)
        
        selected = max(weights, key=weights.get)
        confidence = weights[selected]
        
        gating_decisions[node] = GatingDecision(
            node=node,
            weights=weights,
            selected=selected,
            confidence=confidence,
            features=features,
        )
        
        # Combinación ponderada (en el espacio del producto)
        pos_E = emb_euclidean.positions[node]
        pos_H = emb_hyperbolic.positions[node]
        
        # Normalizar posición hiperbólica para compatibilidad
        # (proyectar al mismo rango que Euclídeo)
        pos_H_normalized = pos_H / (np.linalg.norm(pos_H) + 1e-6) * np.linalg.norm(pos_E + 1e-6)
        
        # Mezcla ponderada
        w_E = weights[GeometrySpace.EUCLIDEAN.value]
        w_H = weights[GeometrySpace.HYPERBOLIC.value]
        mixed_positions[node] = w_E * pos_E + w_H * pos_H_normalized
    
    # 5. Compilar métricas
    euclidean_ratio = sum(
        1 for d in gating_decisions.values() 
        if d.selected == GeometrySpace.EUCLIDEAN.value
    ) / n
    
    avg_confidence = np.mean([d.confidence for d in gating_decisions.values()])
    
    metrics = {
        "n_nodes": n,
        "dim": dim,
        "euclidean_ratio": euclidean_ratio,
        "hyperbolic_ratio": 1 - euclidean_ratio,
        "avg_confidence": avg_confidence,
        "euclidean_stress": emb_euclidean.stress,
        "hyperbolic_loss": emb_hyperbolic.final_loss,
        "hyperbolic_distortion": emb_hyperbolic.distortion,
    }
    
    logger.log_imaginario(
        "moe_embedding_complete",
        details=metrics,
    )
    
    return GeometricMoEResult(
        positions_euclidean=emb_euclidean.positions,
        positions_hyperbolic=emb_hyperbolic.positions,
        gating_decisions=gating_decisions,
        mixed_positions=mixed_positions,
        metrics=metrics,
    )


def compute_mixed_distance(
    moe_result: GeometricMoEResult,
    node_i: int,
    node_j: int,
    hyperbolic_curvature: float = -1.0,
) -> float:
    """
    Calcula distancia ponderada en el espacio mixto.
    
    Usa los pesos de gating de ambos nodos para ponderar
    distancias Euclídea e Hiperbólica.
    """
    decision_i = moe_result.gating_decisions[node_i]
    decision_j = moe_result.gating_decisions[node_j]
    
    # Promediar pesos de ambos nodos
    w_E = (decision_i.weights["E"] + decision_j.weights["E"]) / 2
    w_H = (decision_i.weights["H"] + decision_j.weights["H"]) / 2
    
    # Distancias en cada espacio
    d_E = compute_embedding_distance(moe_result.positions_euclidean, node_i, node_j)
    d_H = compute_hyperbolic_distance(moe_result.positions_hyperbolic, node_i, node_j, hyperbolic_curvature)
    
    # Distancia ponderada
    return w_E * d_E + w_H * d_H


# ============== RAS Integration ==============

@dataclass
class RASState:
    """Estado del sistema RAS para control de MoE."""
    stress_level: float = 0.0  # 0=normal, 1=máximo stress
    budget_remaining: float = 1.0  # Presupuesto computacional
    exploration_rate: float = 0.1  # Tasa de exploración
    
    # Contadores
    escalations: int = 0
    de_escalations: int = 0


def ras_controlled_gating(
    features: NodeFeatures,
    ras_state: RASState,
    base_temperature: float = 1.0,
) -> dict[str, float]:
    """
    Gating controlado por RAS.
    
    El estado RAS modifica el comportamiento del gating:
    - Alto stress → temperatura baja (decisiones más duras)
    - Bajo budget → preferir Euclídeo (más barato)
    - Alta exploración → temperatura alta
    
    Args:
        features: Características del nodo
        ras_state: Estado actual del RAS
        base_temperature: Temperatura base
    
    Returns:
        Pesos de gating modificados por RAS
    """
    # Ajustar temperatura según estado
    effective_temperature = base_temperature * (
        1 + ras_state.exploration_rate - ras_state.stress_level * 0.5
    )
    effective_temperature = max(0.1, effective_temperature)
    
    # Obtener pesos base
    weights = soft_gating(features, temperature=effective_temperature)
    
    # Si bajo presupuesto, sesgar hacia Euclídeo (más rápido)
    if ras_state.budget_remaining < 0.3:
        weights["E"] += 0.2
        weights["H"] -= 0.2
        weights["E"] = max(0, weights["E"])
        weights["H"] = max(0, weights["H"])
        
        # Renormalizar
        total = weights["E"] + weights["H"]
        weights["E"] /= total
        weights["H"] /= total
    
    return weights

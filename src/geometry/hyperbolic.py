"""
Embedding Hiperbólico para grafos (Modelo de Poincaré Ball).

El espacio hiperbólico es ideal para representar estructuras jerárquicas
ya que el volumen crece exponencialmente con el radio (como un árbol).

Implementación ligera sin GPU para prototipado.
Para escalamiento, usar geoopt con PyTorch.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- dim: Dimensionalidad del embedding
- curvature: Curvatura del espacio (c < 0, típicamente -1)
- learning_rate: Tasa de aprendizaje para optimización
- epochs: Número de épocas de entrenamiento
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Callable

from src.verification.rsi_logger import RSILogger


# Constante pequeña para estabilidad numérica
EPS = 1e-6


@dataclass
class HyperbolicEmbeddingResult:
    """Resultado de embedding hiperbólico."""
    positions: dict[int, np.ndarray]  # Coordenadas en Poincaré ball
    dim: int
    curvature: float
    distortion: float
    final_loss: float
    method: str = "poincare"


# ============== Operaciones en el modelo de Poincaré ==============

def poincare_distance(u: np.ndarray, v: np.ndarray, c: float = -1.0) -> float:
    """
    Distancia geodésica en el modelo de Poincaré Ball.
    
    d(u,v) = (1/√|c|) * arcosh(1 + 2|c| * ||u-v||² / ((1-|c|||u||²)(1-|c|||v||²)))
    
    Args:
        u, v: Puntos en el disco de Poincaré (||x|| < 1/√|c|)
        c: Curvatura (negativa, default -1)
    
    Returns:
        Distancia geodésica
    """
    c_abs = abs(c)
    sqrt_c = np.sqrt(c_abs)
    
    # Normas al cuadrado
    u_sq = np.sum(u ** 2)
    v_sq = np.sum(v ** 2)
    uv_sq = np.sum((u - v) ** 2)
    
    # Denominadores
    denom_u = 1 - c_abs * u_sq
    denom_v = 1 - c_abs * v_sq
    
    # Evitar división por cero
    denom_u = max(denom_u, EPS)
    denom_v = max(denom_v, EPS)
    
    # Argumento de arcosh
    x = 1 + 2 * c_abs * uv_sq / (denom_u * denom_v)
    x = max(x, 1 + EPS)  # arcosh(x) requiere x >= 1
    
    return (1 / sqrt_c) * np.arccosh(x)


def project_to_ball(x: np.ndarray, c: float = -1.0, max_norm: float = 0.99) -> np.ndarray:
    """
    Proyecta un punto al interior del disco de Poincaré.
    
    Args:
        x: Punto a proyectar
        c: Curvatura
        max_norm: Norma máxima relativa al radio
    
    Returns:
        Punto proyectado
    """
    c_abs = abs(c)
    radius = 1 / np.sqrt(c_abs)
    max_allowed = radius * max_norm
    
    norm = np.linalg.norm(x)
    if norm > max_allowed:
        x = x * max_allowed / (norm + EPS)
    
    return x


def mobius_add(u: np.ndarray, v: np.ndarray, c: float = -1.0) -> np.ndarray:
    """
    Suma de Möbius (adición en espacio hiperbólico).
    
    u ⊕ v = ((1 + 2c<u,v> + c||v||²)u + (1 - c||u||²)v) / (1 + 2c<u,v> + c²||u||²||v||²)
    """
    c_abs = abs(c)
    
    u_sq = np.sum(u ** 2)
    v_sq = np.sum(v ** 2)
    uv = np.sum(u * v)
    
    num_u = (1 + 2 * c_abs * uv + c_abs * v_sq) * u
    num_v = (1 - c_abs * u_sq) * v
    denom = 1 + 2 * c_abs * uv + c_abs ** 2 * u_sq * v_sq
    
    return (num_u + num_v) / (denom + EPS)


def exp_map(v: np.ndarray, x: np.ndarray = None, c: float = -1.0) -> np.ndarray:
    """
    Mapa exponencial: tangente → manifold.
    
    Mueve desde x en dirección v (en el espacio tangente en x).
    Si x es None, usa el origen.
    """
    c_abs = abs(c)
    sqrt_c = np.sqrt(c_abs)
    
    if x is None:
        x = np.zeros_like(v)
    
    v_norm = np.linalg.norm(v)
    if v_norm < EPS:
        return x
    
    # Lambda_x = 1 / (1 - c|x|²)
    x_sq = np.sum(x ** 2)
    lambda_x = 1 / (1 - c_abs * x_sq + EPS)
    
    # exp_x(v) = x ⊕ (tanh(√c * λ_x * ||v|| / 2) * v / (√c * ||v||))
    coeff = np.tanh(sqrt_c * lambda_x * v_norm / 2) / (sqrt_c * v_norm + EPS)
    
    return mobius_add(x, coeff * v, c)


def log_map(y: np.ndarray, x: np.ndarray = None, c: float = -1.0) -> np.ndarray:
    """
    Mapa logarítmico: manifold → tangente.
    
    Retorna el vector tangente en x que apunta hacia y.
    """
    c_abs = abs(c)
    sqrt_c = np.sqrt(c_abs)
    
    if x is None:
        x = np.zeros_like(y)
    
    # -x ⊕ y
    neg_x = -x
    diff = mobius_add(neg_x, y, c)
    
    diff_norm = np.linalg.norm(diff)
    if diff_norm < EPS:
        return np.zeros_like(y)
    
    # Lambda_x
    x_sq = np.sum(x ** 2)
    lambda_x = 1 / (1 - c_abs * x_sq + EPS)
    
    # 2 / (√c * λ_x) * arctanh(√c * ||diff||) * diff / ||diff||
    coeff = (2 / (sqrt_c * lambda_x + EPS)) * np.arctanh(min(sqrt_c * diff_norm, 1 - EPS))
    
    return coeff * diff / (diff_norm + EPS)


# ============== Embedding hiperbólico ==============

def poincare_embedding(
    G: nx.Graph,
    dim: int = 32,  # PARAM
    curvature: float = -1.0,  # PARAM
    learning_rate: float = 0.1,  # PARAM
    epochs: int = 100,  # PARAM
    burn_in: float = 0.1,  # PARAM: fracción inicial con lr bajo
    seed: int | None = None,
    logger: RSILogger | None = None,
) -> HyperbolicEmbeddingResult:
    """
    Embedding en el modelo de Poincaré Ball usando RSGD.
    
    Optimiza para que aristas conecten nodos cercanos en el espacio hiperbólico.
    Ideal para grafos con estructura jerárquica.
    
    Args:
        G: Grafo
        dim: Dimensionalidad
        curvature: Curvatura (negativa)
        learning_rate: Tasa de aprendizaje
        epochs: Número de épocas
        burn_in: Fracción de épocas con lr reducido al inicio
        seed: Semilla
        logger: RSI logger
    
    Returns:
        HyperbolicEmbeddingResult
    """
    if logger is None:
        logger = RSILogger("hyperbolic", console_output=False)
    
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Inicialización cerca del origen (uniforme en disco pequeño)
    c_abs = abs(curvature)
    init_radius = 0.1 / np.sqrt(c_abs)
    embeddings = rng.uniform(-init_radius, init_radius, (n, dim))
    
    # Proyectar al ball
    for i in range(n):
        embeddings[i] = project_to_ball(embeddings[i], curvature)
    
    # Preparar datos de entrenamiento
    edges = list(G.edges())
    n_edges = len(edges)
    
    # Épocas con burn-in
    burn_in_epochs = int(epochs * burn_in)
    
    losses = []
    
    for epoch in range(epochs):
        # Learning rate schedule
        if epoch < burn_in_epochs:
            lr = learning_rate * 0.1
        else:
            lr = learning_rate
        
        epoch_loss = 0.0
        rng.shuffle(edges)
        
        for u, v in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            
            x_i = embeddings[i]
            x_j = embeddings[j]
            
            # Distancia actual
            d = poincare_distance(x_i, x_j, curvature)
            
            # Pérdida: queremos minimizar distancia para aristas
            # Usamos d² como pérdida simple
            epoch_loss += d ** 2
            
            # Gradiente en espacio tangente (aproximación)
            # Negativo del log_map da la dirección de acercamiento
            grad_i = log_map(x_j, x_i, curvature)
            grad_j = log_map(x_i, x_j, curvature)
            
            # Actualización (RSGD: gradient descent en manifold)
            x_i_new = exp_map(-lr * grad_i / (np.linalg.norm(grad_i) + EPS), x_i, curvature)
            x_j_new = exp_map(-lr * grad_j / (np.linalg.norm(grad_j) + EPS), x_j, curvature)
            
            # Proyectar al ball
            embeddings[i] = project_to_ball(x_i_new, curvature)
            embeddings[j] = project_to_ball(x_j_new, curvature)
        
        epoch_loss /= max(n_edges, 1)
        losses.append(epoch_loss)
        
        # Log cada 20% del progreso
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.log_simbolico(
                "training_progress",
                details={"epoch": epoch + 1, "epochs": epochs},
                metrics={"loss": epoch_loss},
            )
    
    # Construir resultado
    positions = {nodes[i]: embeddings[i] for i in range(n)}
    
    # Calcular distorsión
    distortion = compute_hyperbolic_distortion(G, positions, curvature)
    
    logger.log_imaginario(
        "poincare_embedding_created",
        details={"n_nodes": n, "dim": dim, "curvature": curvature},
        metrics={"final_loss": losses[-1] if losses else 0, "distortion": distortion},
    )
    
    return HyperbolicEmbeddingResult(
        positions=positions,
        dim=dim,
        curvature=curvature,
        distortion=distortion,
        final_loss=losses[-1] if losses else 0,
    )


def compute_hyperbolic_distortion(
    G: nx.Graph,
    positions: dict[int, np.ndarray],
    curvature: float = -1.0,
) -> float:
    """
    Calcula la distorsión del embedding hiperbólico.
    
    Distorsión = max(d_hyp / d_graph, d_graph / d_hyp) sobre pares conectados.
    """
    nodes = list(G.nodes())
    
    # Solo calcular sobre muestra para grafos grandes
    max_pairs = 1000
    edges = list(G.edges())
    
    if len(edges) > max_pairs:
        rng = np.random.default_rng(42)
        edges = list(rng.choice(edges, size=max_pairs, replace=False))
    
    ratios = []
    for u, v in edges:
        d_graph = 1.0  # Aristas tienen distancia 1
        d_hyp = poincare_distance(positions[u], positions[v], curvature)
        
        if d_hyp > EPS:
            ratios.append(max(d_graph / d_hyp, d_hyp / d_graph))
    
    return max(ratios) if ratios else 1.0


def embed_hyperbolic(
    G: nx.Graph,
    dim: int = 32,
    curvature: float = -1.0,
    **kwargs,
) -> HyperbolicEmbeddingResult:
    """
    Factory function para embedding hiperbólico.
    
    Args:
        G: Grafo
        dim: Dimensionalidad
        curvature: Curvatura del espacio
        **kwargs: Argumentos adicionales para poincare_embedding
    
    Returns:
        HyperbolicEmbeddingResult
    """
    return poincare_embedding(G, dim=dim, curvature=curvature, **kwargs)


def compute_hyperbolic_distance(
    positions: dict[int, np.ndarray],
    node_i: int,
    node_j: int,
    curvature: float = -1.0,
) -> float:
    """Distancia hiperbólica entre dos nodos en el embedding."""
    return poincare_distance(positions[node_i], positions[node_j], curvature)

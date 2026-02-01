"""
Embedding Euclídeo para grafos.

Este módulo implementa embeddings en espacio Euclídeo (R^n) usando
técnicas clásicas como MDS, Laplacian Eigenmaps, y Node2Vec simplificado.

Sirve como baseline para comparar con embeddings hiperbólicos.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- dim: Dimensionalidad del embedding
- method: Algoritmo de embedding
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
from typing import Literal
from dataclasses import dataclass

from src.verification.rsi_logger import RSILogger


@dataclass
class EmbeddingResult:
    """Resultado de un embedding."""
    positions: dict[int, np.ndarray]  # {node: coords}
    dim: int
    method: str
    distortion: float  # Medida de calidad
    stress: float  # Error de reconstrucción


def compute_distance_matrix(G: nx.Graph, weight: str | None = None) -> np.ndarray:
    """
    Calcula matriz de distancias shortest-path para el grafo.
    
    Args:
        G: Grafo
        weight: Nombre del atributo de peso
    
    Returns:
        Matriz n×n de distancias
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    
    for source in nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        for target, length in lengths.items():
            i, j = node_to_idx[source], node_to_idx[target]
            D[i, j] = length
    
    return D, nodes


def mds_embedding(
    G: nx.Graph,
    dim: int = 32,  # PARAM: dimensionalidad
    weight: str | None = None,
    logger: RSILogger | None = None,
) -> EmbeddingResult:
    """
    Embedding usando Multidimensional Scaling (MDS) clásico.
    
    MDS intenta preservar distancias de grafo en el espacio Euclídeo.
    
    Args:
        G: Grafo
        dim: Número de dimensiones del embedding
        weight: Nombre del atributo de peso
        logger: RSI logger
    
    Returns:
        EmbeddingResult con posiciones y métricas
    """
    if logger is None:
        logger = RSILogger("euclidean", console_output=False)
    
    # Matriz de distancias
    D, nodes = compute_distance_matrix(G, weight)
    n = len(nodes)
    
    # Manejar infinitos (nodos desconectados)
    max_finite = D[D != np.inf].max() if np.any(D != np.inf) else 1
    D[D == np.inf] = max_finite * 2
    
    # MDS clásico: doble centrado
    D_sq = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J
    
    # Eigendecomposition
    # Usar min(dim, n-1) para evitar problemas con grafos pequeños
    k = min(dim, n - 1)
    try:
        eigenvalues, eigenvectors = eigsh(B, k=k, which='LA')
    except Exception:
        # Fallback a eigendecomposition completa
        eigenvalues, eigenvectors = linalg.eigh(B)
        eigenvalues = eigenvalues[-k:]
        eigenvectors = eigenvectors[:, -k:]
    
    # Ordenar por eigenvalue descendente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Solo usar eigenvalues positivos
    positive_mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[positive_mask]
    eigenvectors = eigenvectors[:, positive_mask]
    
    # Calcular coordenadas
    if len(eigenvalues) == 0:
        # Fallback: posiciones aleatorias
        coords = np.random.randn(n, dim) * 0.01
    else:
        coords = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        # Pad con ceros si hay menos dimensiones
        if coords.shape[1] < dim:
            padding = np.zeros((n, dim - coords.shape[1]))
            coords = np.hstack([coords, padding])
    
    # Crear diccionario de posiciones
    positions = {nodes[i]: coords[i, :dim] for i in range(n)}
    
    # Calcular distorsión (stress) con mejor estabilidad numérica
    D_embedded = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D_embedded[i, j] = d
            D_embedded[j, i] = d
    
    # Stress normalizado (Kruskal's stress-1)
    mask = (D < np.inf) & (D > 0)
    if mask.sum() > 0:
        numerator = np.sum((D[mask] - D_embedded[mask]) ** 2)
        denominator = np.sum(D[mask] ** 2)
        stress = np.sqrt(numerator / (denominator + 1e-10))
    else:
        stress = 0.0
    
    # Distorsión: percentil 95 del ratio para evitar outliers
    # (mejor que max absoluto que puede ser un solo par problemático)
    ratios = []
    min_dist_threshold = 1e-6  # Evitar divisiones por cero
    for i in range(n):
        for j in range(i + 1, n):
            d_orig = D[i, j]
            d_emb = D_embedded[i, j]
            if d_orig > 0 and d_orig < np.inf and d_emb > min_dist_threshold:
                ratio = max(d_orig / d_emb, d_emb / d_orig)
                if np.isfinite(ratio):
                    ratios.append(ratio)
    
    if ratios:
        distortion = float(np.percentile(ratios, 95))  # Percentil 95 en vez de max
    else:
        distortion = 1.0
    
    logger.log_imaginario(
        "mds_embedding_created",
        details={"n_nodes": n, "dim": dim, "n_positive_eigenvalues": len(eigenvalues)},
        metrics={"stress": stress, "distortion": distortion},
    )
    
    return EmbeddingResult(
        positions=positions,
        dim=dim,
        method="mds",
        distortion=distortion,
        stress=stress,
    )


def laplacian_embedding(
    G: nx.Graph,
    dim: int = 32,  # PARAM: dimensionalidad
    normalized: bool = True,  # PARAM: normalizar Laplaciano
    logger: RSILogger | None = None,
) -> EmbeddingResult:
    """
    Embedding usando Laplacian Eigenmaps.
    
    Usa los eigenvectores más pequeños del Laplaciano normalizado.
    Captura estructura de clusters/comunidades.
    
    Args:
        G: Grafo
        dim: Número de dimensiones
        normalized: Usar Laplaciano normalizado
        logger: RSI logger
    
    Returns:
        EmbeddingResult
    """
    if logger is None:
        logger = RSILogger("euclidean", console_output=False)
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Calcular Laplaciano
    if normalized:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()
    
    # Eigendecomposition (eigenvectors más pequeños, excluyendo el primero)
    k = min(dim + 1, n)
    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
    except Exception:
        eigenvalues, eigenvectors = linalg.eigh(L)
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    
    # Ordenar por eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    
    # Saltar el primer eigenvector (constante)
    coords = eigenvectors[:, 1:dim + 1]
    
    # Pad si necesario
    if coords.shape[1] < dim:
        padding = np.zeros((n, dim - coords.shape[1]))
        coords = np.hstack([coords, padding])
    
    positions = {nodes[i]: coords[i] for i in range(n)}
    
    # Calcular métricas (simplificado)
    D, _ = compute_distance_matrix(G)
    D_embedded = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D_embedded[i, j] = d
            D_embedded[j, i] = d
    
    mask = (D != np.inf) & (D > 0)
    if mask.sum() > 0:
        stress = np.sqrt(np.sum((D[mask] - D_embedded[mask]) ** 2) / np.sum(D[mask] ** 2))
    else:
        stress = 0.0
    
    logger.log_imaginario(
        "laplacian_embedding_created",
        details={"n_nodes": n, "dim": dim, "normalized": normalized},
        metrics={"stress": stress},
    )
    
    return EmbeddingResult(
        positions=positions,
        dim=dim,
        method="laplacian",
        distortion=1.0,  # No calculamos distorsión para Laplacian
        stress=stress,
    )


def spring_layout_embedding(
    G: nx.Graph,
    dim: int = 2,  # PARAM: solo 2 o 3 para spring layout
    iterations: int = 50,  # PARAM: iteraciones
    seed: int | None = None,
    logger: RSILogger | None = None,
) -> EmbeddingResult:
    """
    Embedding usando force-directed layout (Fruchterman-Reingold).
    
    Bueno para visualización pero limitado a 2-3 dimensiones.
    
    Args:
        G: Grafo
        dim: Número de dimensiones (2 o 3)
        iterations: Número de iteraciones
        seed: Semilla
        logger: RSI logger
    
    Returns:
        EmbeddingResult
    """
    if logger is None:
        logger = RSILogger("euclidean", console_output=False)
    
    if dim > 3:
        dim = 3  # NetworkX solo soporta hasta 3D
    
    pos = nx.spring_layout(G, dim=dim, iterations=iterations, seed=seed)
    
    # Convertir a arrays
    positions = {node: np.array(coords) for node, coords in pos.items()}
    
    logger.log_imaginario(
        "spring_embedding_created",
        details={"n_nodes": G.number_of_nodes(), "dim": dim},
    )
    
    return EmbeddingResult(
        positions=positions,
        dim=dim,
        method="spring",
        distortion=1.0,
        stress=0.0,  # No calculamos para spring
    )


def embed_euclidean(
    G: nx.Graph,
    dim: int = 32,
    method: Literal["mds", "laplacian", "spring"] = "mds",  # PARAM
    **kwargs,
) -> EmbeddingResult:
    """
    Factory function para embeddings Euclídeos.
    
    Args:
        G: Grafo
        dim: Dimensionalidad
        method: Método de embedding
        **kwargs: Argumentos adicionales para el método
    
    Returns:
        EmbeddingResult
    """
    methods = {
        "mds": mds_embedding,
        "laplacian": laplacian_embedding,
        "spring": spring_layout_embedding,
    }
    
    if method not in methods:
        raise ValueError(f"Método desconocido: {method}. Opciones: {list(methods.keys())}")
    
    return methods[method](G, dim=dim, **kwargs)


def compute_embedding_distance(
    positions: dict[int, np.ndarray],
    node_i: int,
    node_j: int,
) -> float:
    """Distancia Euclídea entre dos nodos en el embedding."""
    return float(np.linalg.norm(positions[node_i] - positions[node_j]))

"""
Routing y métricas en grafos.

Este módulo implementa greedy routing y métricas para evaluar
el rendimiento de routing antes/después de Ricci-cleaning.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- n_queries: Número de queries para evaluar routing
- seed: Semilla para reproducibilidad
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from typing import Literal
from dataclasses import dataclass

from src.verification.rsi_logger import RSILogger


@dataclass
class RoutingResult:
    """Resultado de una query de routing."""
    source: int
    target: int
    path: list[int] | None  # None si no hay ruta
    hops: int  # -1 si no hay ruta
    success: bool
    oracle_hops: int  # Camino más corto
    stretch: float  # hops / oracle_hops


def greedy_routing(
    G: nx.Graph,
    source: int,
    target: int,
    pos: dict[int, np.ndarray] | None = None,  # Posiciones para distancia
    max_hops: int | None = None,  # PARAM: límite de saltos
    distance_func: Literal["hop", "euclidean"] = "hop",  # PARAM
) -> RoutingResult:
    """
    Ejecuta greedy routing desde source a target.
    
    Greedy routing: en cada paso, elegir el vecino más cercano al destino.
    
    Distancias disponibles:
    - "hop": Usar distancia de grafo (BFS hacia adelante)
    - "euclidean": Usar posiciones euclídeas (requiere pos)
    
    Args:
        G: Grafo
        source: Nodo origen
        target: Nodo destino
        pos: Diccionario de posiciones {nodo: np.array([x, y, ...])}
        max_hops: Límite de saltos (default: n_nodes)
        distance_func: Función de distancia a usar
    
    Returns:
        RoutingResult con información del routing
    """
    if source == target:
        # Obtener distancia oracle
        try:
            oracle = nx.shortest_path_length(G, source, target)
        except nx.NetworkXNoPath:
            oracle = -1
        
        return RoutingResult(
            source=source,
            target=target,
            path=[source],
            hops=0,
            success=True,
            oracle_hops=oracle,
            stretch=1.0,
        )
    
    if max_hops is None:
        max_hops = G.number_of_nodes()
    
    # Función de distancia al target
    if distance_func == "euclidean":
        if pos is None:
            raise ValueError("Posiciones requeridas para distancia euclidiana")
        def dist_to_target(node):
            return np.linalg.norm(pos[node] - pos[target])
    else:  # hop
        # Pre-calcular distancias desde target (BFS inverso)
        distances_from_target = nx.single_source_shortest_path_length(G, target)
        def dist_to_target(node):
            return distances_from_target.get(node, float('inf'))
    
    # Greedy routing
    current = source
    path = [current]
    visited = {current}
    
    while current != target and len(path) <= max_hops:
        neighbors = [n for n in G.neighbors(current) if n not in visited]
        
        if not neighbors:
            # Dead end
            break
        
        # Elegir vecino más cercano al target
        best_neighbor = min(neighbors, key=dist_to_target)
        
        # Verificar progreso (evitar loops)
        if dist_to_target(best_neighbor) >= dist_to_target(current):
            # No hay progreso, intentar cualquier vecino no visitado
            pass  # Continuamos con el mejor encontrado
        
        current = best_neighbor
        path.append(current)
        visited.add(current)
    
    success = current == target
    hops = len(path) - 1 if success else -1
    
    # Oracle path
    try:
        oracle_hops = nx.shortest_path_length(G, source, target)
    except nx.NetworkXNoPath:
        oracle_hops = -1
    
    stretch = hops / oracle_hops if success and oracle_hops > 0 else float('inf')
    
    return RoutingResult(
        source=source,
        target=target,
        path=path if success else None,
        hops=hops,
        success=success,
        oracle_hops=oracle_hops,
        stretch=stretch,
    )


def evaluate_routing(
    G: nx.Graph,
    n_queries: int = 100,  # PARAM: número de queries
    seed: int | None = None,  # PARAM: semilla
    pos: dict[int, np.ndarray] | None = None,
    distance_func: Literal["hop", "euclidean"] = "hop",
    logger: RSILogger | None = None,
) -> dict:
    """
    Evalúa el rendimiento de routing en un grafo.
    
    Ejecuta n_queries de routing entre pares aleatorios y calcula métricas.
    
    Args:
        G: Grafo a evaluar
        n_queries: Número de queries aleatorias
        seed: Semilla para reproducibilidad
        pos: Posiciones para distancia euclidiana
        distance_func: Función de distancia
        logger: RSI logger
    
    Returns:
        Diccionario con métricas de routing
    """
    if logger is None:
        logger = RSILogger("routing", console_output=False)
    
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n < 2:
        return {"error": "Grafo demasiado pequeño"}
    
    results = []
    
    for _ in range(n_queries):
        # Elegir par aleatorio
        source, target = rng.choice(nodes, size=2, replace=False)
        
        result = greedy_routing(
            G, source, target, 
            pos=pos, 
            distance_func=distance_func,
        )
        results.append(result)
    
    # Calcular métricas
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    
    success_rate = len(successes) / len(results)
    
    if successes:
        stretches = [r.stretch for r in successes if r.stretch != float('inf')]
        avg_stretch = np.mean(stretches) if stretches else float('inf')
        median_stretch = np.median(stretches) if stretches else float('inf')
        max_stretch = np.max(stretches) if stretches else float('inf')
        
        hops = [r.hops for r in successes]
        avg_hops = np.mean(hops)
    else:
        avg_stretch = float('inf')
        median_stretch = float('inf')
        max_stretch = float('inf')
        avg_hops = -1
    
    metrics = {
        "n_queries": n_queries,
        "success_rate": success_rate,
        "n_successes": len(successes),
        "n_failures": len(failures),
        "avg_stretch": float(avg_stretch),
        "median_stretch": float(median_stretch),
        "max_stretch": float(max_stretch) if max_stretch != float('inf') else -1,
        "avg_hops": float(avg_hops),
        "distance_func": distance_func,
    }
    
    logger.log_simbolico(
        "routing_evaluated",
        details={"n_nodes": n, "n_edges": G.number_of_edges()},
        metrics=metrics,
    )
    
    return metrics


def compare_routing(
    G_before: nx.Graph,
    G_after: nx.Graph,
    n_queries: int = 100,
    seed: int | None = None,
    logger: RSILogger | None = None,
) -> dict:
    """
    Compara routing antes y después de una transformación (e.g., Ricci-cleaning).
    
    Usa las mismas queries en ambos grafos para comparación justa.
    
    Args:
        G_before: Grafo antes de transformación
        G_after: Grafo después de transformación
        n_queries: Número de queries
        seed: Semilla (misma para ambos)
        logger: RSI logger
    
    Returns:
        Diccionario con métricas comparativas
    """
    if logger is None:
        logger = RSILogger("routing", console_output=False)
    
    metrics_before = evaluate_routing(G_before, n_queries, seed, logger=logger)
    metrics_after = evaluate_routing(G_after, n_queries, seed, logger=logger)
    
    # Calcular deltas
    delta_success = metrics_after["success_rate"] - metrics_before["success_rate"]
    delta_stretch = metrics_before["avg_stretch"] - metrics_after["avg_stretch"]  # Negativo = mejor
    
    comparison = {
        "before": metrics_before,
        "after": metrics_after,
        "delta_success_rate": delta_success,
        "delta_stretch": delta_stretch,
        "improvement": delta_success > 0 or (delta_success == 0 and delta_stretch > 0),
    }
    
    logger.log_simbolico(
        "routing_comparison",
        details=comparison,
        metrics={
            "delta_success_rate": delta_success,
            "delta_stretch": delta_stretch,
        },
    )
    
    return comparison


def compute_graph_metrics(G: nx.Graph) -> dict:
    """
    Calcula métricas básicas del grafo.
    
    Args:
        G: Grafo
    
    Returns:
        Diccionario con métricas
    """
    metrics = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": nx.number_connected_components(G),
        "density": nx.density(G),
    }
    
    if nx.is_connected(G):
        metrics["diameter"] = nx.diameter(G)
        metrics["avg_path_length"] = nx.average_shortest_path_length(G)
    else:
        # Calcular para el componente más grande
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc).copy()
        metrics["largest_component_size"] = len(largest_cc)
        if len(largest_cc) > 1:
            metrics["diameter_largest"] = nx.diameter(G_largest)
            metrics["avg_path_length_largest"] = nx.average_shortest_path_length(G_largest)
    
    # Clustering
    metrics["avg_clustering"] = nx.average_clustering(G)
    
    return metrics

"""
Ricci-cleaning: limpieza de grafos basada en curvatura.

Este módulo implementa la eliminación/atenuación de aristas con
curvatura negativa (candidatas a ruido o atajos indeseados).

VARIABLES AJUSTABLES (marcadas con # PARAM):
- threshold: Umbral de curvatura para eliminar aristas
- preserve_connectivity: Mantener grafo conexo
- soft_clean: Atenuar en lugar de eliminar
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from typing import Literal

from src.core.curvature import compute_forman_ricci
from src.verification.rsi_logger import RSILogger


def is_bridge(G: nx.Graph, edge: tuple) -> bool:
    """
    Verifica si una arista es un puente (su eliminación desconecta el grafo).
    
    Args:
        G: Grafo
        edge: Arista a verificar (u, v)
    
    Returns:
        True si la arista es un puente
    """
    u, v = edge
    # Verificar si existe un camino alternativo
    G_temp = G.copy()
    G_temp.remove_edge(u, v)
    return not nx.has_path(G_temp, u, v)


def ricci_clean(
    G: nx.Graph,
    threshold: float = -0.5,  # PARAM: umbral de curvatura
    preserve_connectivity: bool = True,  # PARAM: mantener conexo
    curvatures: dict[tuple, float] | None = None,  # Pre-calculadas
    weight: str | None = None,
    logger: RSILogger | None = None,
) -> tuple[nx.Graph, dict]:
    """
    Limpia un grafo eliminando aristas con curvatura muy negativa.
    
    Aristas con curvatura < threshold son removidas (si no son puentes
    y preserve_connectivity=True).
    
    Args:
        G: Grafo original
        threshold: Umbral de curvatura (aristas < threshold se eliminan)
        preserve_connectivity: Si True, no elimina aristas puente
        curvatures: Diccionario de curvaturas pre-calculadas (opcional)
        weight: Nombre del atributo de peso
        logger: RSI logger
    
    Returns:
        Tupla (grafo_limpio, info)
        donde info contiene estadísticas de la limpieza
    """
    if logger is None:
        logger = RSILogger("ricci_cleaning", console_output=False)
    
    # Calcular curvaturas si no se proporcionan
    if curvatures is None:
        curvatures = compute_forman_ricci(G, weight=weight, logger=logger)
    
    # Registrar contrato de threshold
    logger.log_simbolico(
        "threshold_contract",
        details={"threshold": threshold, "preserve_connectivity": preserve_connectivity},
    )
    
    # Identificar aristas candidatas a eliminar
    candidates = [
        edge for edge, curv in curvatures.items()
        if curv < threshold
    ]
    
    # Crear copia del grafo
    G_clean = G.copy()
    
    removed = []
    preserved_bridges = []
    
    for edge in candidates:
        # Normalizar dirección de arista
        u, v = edge
        if not G_clean.has_edge(u, v):
            continue
        
        if preserve_connectivity and is_bridge(G_clean, (u, v)):
            preserved_bridges.append(edge)
            logger.log_real(
                "bridge_preserved",
                details={"edge": str(edge), "curvature": curvatures[edge]},
            )
        else:
            G_clean.remove_edge(u, v)
            removed.append(edge)
    
    # Verificar conectividad resultante
    n_components_before = nx.number_connected_components(G)
    n_components_after = nx.number_connected_components(G_clean)
    
    if n_components_after > n_components_before:
        logger.log_real(
            "connectivity_degraded",
            details={
                "components_before": n_components_before,
                "components_after": n_components_after,
            },
        )
    
    # Compilar info
    info = {
        "edges_before": G.number_of_edges(),
        "edges_after": G_clean.number_of_edges(),
        "edges_removed": len(removed),
        "bridges_preserved": len(preserved_bridges),
        "threshold": threshold,
        "components_before": n_components_before,
        "components_after": n_components_after,
        "removed_edges": removed,
    }
    
    logger.log_simbolico(
        "cleaning_completed",
        details=info,
        metrics={
            "removal_rate": len(removed) / max(1, G.number_of_edges()),
            "edges_removed": len(removed),
        },
    )
    
    return G_clean, info


def ricci_clean_soft(
    G: nx.Graph,
    threshold: float = -0.5,  # PARAM: umbral
    attenuation_factor: float = 0.5,  # PARAM: factor de atenuación
    curvatures: dict[tuple, float] | None = None,
    weight: str = "weight",
    logger: RSILogger | None = None,
) -> tuple[nx.Graph, dict]:
    """
    Limpieza suave: atenúa el peso de aristas con curvatura negativa
    en lugar de eliminarlas.
    
    Útil cuando queremos preservar conectividad pero reducir influencia
    de aristas sospechosas.
    
    Args:
        G: Grafo original
        threshold: Umbral de curvatura
        attenuation_factor: Factor para reducir peso (0-1)
        curvatures: Curvaturas pre-calculadas
        weight: Nombre del atributo de peso
        logger: RSI logger
    
    Returns:
        Tupla (grafo_atenuado, info)
    """
    if logger is None:
        logger = RSILogger("ricci_cleaning", console_output=False)
    
    if curvatures is None:
        curvatures = compute_forman_ricci(G, logger=logger)
    
    G_soft = G.copy()
    
    attenuated = []
    
    for edge, curv in curvatures.items():
        u, v = edge
        if curv < threshold and G_soft.has_edge(u, v):
            # Obtener peso actual o default 1.0
            current_weight = G_soft[u][v].get(weight, 1.0)
            new_weight = current_weight * attenuation_factor
            G_soft[u][v][weight] = new_weight
            attenuated.append(edge)
    
    info = {
        "edges_attenuated": len(attenuated),
        "attenuation_factor": attenuation_factor,
        "threshold": threshold,
    }
    
    logger.log_simbolico(
        "soft_cleaning_completed",
        details=info,
        metrics={"attenuation_rate": len(attenuated) / max(1, G.number_of_edges())},
    )
    
    return G_soft, info


def suggest_threshold(
    curvatures: dict[tuple, float],
    method: Literal["percentile", "std"] = "percentile",  # PARAM
    percentile: int = 10,  # PARAM: percentil para método percentile
    n_std: float = 2.0,  # PARAM: número de std para método std
) -> float:
    """
    Sugiere un umbral de curvatura basado en la distribución.
    
    Métodos:
    - "percentile": Usa el percentil N de la distribución
    - "std": Usa mean - n_std * std
    
    Args:
        curvatures: Diccionario de curvaturas
        method: Método para calcular umbral
        percentile: Percentil a usar (método percentile)
        n_std: Número de desviaciones estándar (método std)
    
    Returns:
        Umbral sugerido
    """
    values = np.array(list(curvatures.values()))
    
    if method == "percentile":
        return float(np.percentile(values, percentile))
    elif method == "std":
        return float(np.mean(values) - n_std * np.std(values))
    else:
        raise ValueError(f"Método desconocido: {method}")

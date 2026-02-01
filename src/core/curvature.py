"""
Cálculo de curvatura discreta en grafos.

Este módulo implementa Forman-Ricci curvature, una medida de curvatura
discreta para aristas en grafos que captura propiedades geométricas locales.

Referencia:
- Forman, R. (2003) "Bochner's Method for Cell Complexes and Combinatorial Ricci Curvature"

VARIABLES AJUSTABLES (marcadas con # PARAM):
- curvature_type: "forman" o "ollivier" (futuro)
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from typing import Literal

from src.verification.rsi_logger import RSILogger, get_logger


def compute_forman_ricci(
    G: nx.Graph,
    weight: str | None = None,  # PARAM: nombre del atributo de peso
    logger: RSILogger | None = None,
) -> dict[tuple, float]:
    """
    Calcula la curvatura Forman-Ricci para cada arista del grafo.
    
    La curvatura Forman-Ricci para una arista e=(u,v) se define como:
    
        F(e) = w(e) * (
            2/w(u) + 2/w(v) 
            - sum_{e' ~ u, e' != e} w(e)/sqrt(w(e)*w(e'))
            - sum_{e' ~ v, e' != e} w(e)/sqrt(w(e)*w(e'))
        )
    
    Para grafos sin peso, esto se simplifica a:
    
        F(e) = 4 - deg(u) - deg(v)
    
    Donde deg(x) es el grado del nodo x.
    
    Interpretación:
    - F > 0: Curvatura positiva (región "esférica", conexión fuerte)
    - F = 0: Curvatura cero (región "plana")
    - F < 0: Curvatura negativa (región "hiperbólica", posible ruido/atajo)
    
    Args:
        G: Grafo NetworkX
        weight: Nombre del atributo de peso de aristas (None = sin peso)
        logger: RSI logger opcional para trazabilidad
    
    Returns:
        Diccionario {(u, v): curvatura} para cada arista
    """
    if logger is None:
        logger = RSILogger("curvature", console_output=False)
    
    curvatures = {}
    
    if weight is None:
        # Caso sin peso: F(e) = 4 - deg(u) - deg(v)
        for u, v in G.edges():
            deg_u = G.degree(u)
            deg_v = G.degree(v)
            curvatures[(u, v)] = 4 - deg_u - deg_v
    else:
        # Caso con peso: fórmula completa
        for u, v in G.edges():
            edge_weight = G[u][v].get(weight, 1.0)
            
            # Peso del nodo = suma de pesos de aristas incidentes
            node_weight_u = sum(G[u][n].get(weight, 1.0) for n in G.neighbors(u))
            node_weight_v = sum(G[v][n].get(weight, 1.0) for n in G.neighbors(v))
            
            # Término de nodos
            node_term = 2.0 / node_weight_u + 2.0 / node_weight_v
            
            # Términos de aristas vecinas
            edge_term_u = sum(
                edge_weight / np.sqrt(edge_weight * G[u][n].get(weight, 1.0))
                for n in G.neighbors(u) if n != v
            )
            edge_term_v = sum(
                edge_weight / np.sqrt(edge_weight * G[v][n].get(weight, 1.0))
                for n in G.neighbors(v) if n != u
            )
            
            curvatures[(u, v)] = edge_weight * (node_term - edge_term_u - edge_term_v)
    
    # Log de estadísticas
    curv_values = list(curvatures.values())
    stats = {
        "n_edges": len(curvatures),
        "min": float(np.min(curv_values)),
        "max": float(np.max(curv_values)),
        "mean": float(np.mean(curv_values)),
        "std": float(np.std(curv_values)),
        "n_positive": sum(1 for c in curv_values if c > 0),
        "n_negative": sum(1 for c in curv_values if c < 0),
        "n_zero": sum(1 for c in curv_values if c == 0),
    }
    
    logger.log_simbolico(
        "forman_ricci_computed",
        details={"weighted": weight is not None},
        metrics=stats,
    )
    
    return curvatures


def get_curvature_distribution(
    curvatures: dict[tuple, float],
    bins: int = 20,  # PARAM: número de bins para histograma
) -> dict[str, any]:
    """
    Analiza la distribución de curvaturas.
    
    Args:
        curvatures: Diccionario de curvaturas por arista
        bins: Número de bins para el histograma
    
    Returns:
        Diccionario con estadísticas y distribución
    """
    values = np.array(list(curvatures.values()))
    
    hist, bin_edges = np.histogram(values, bins=bins)
    
    return {
        "count": len(values),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
    }


def classify_edges_by_curvature(
    curvatures: dict[tuple, float],
    negative_threshold: float = -0.5,  # PARAM: umbral para "muy negativo"
    positive_threshold: float = 0.5,  # PARAM: umbral para "muy positivo"
) -> dict[str, list[tuple]]:
    """
    Clasifica aristas según su curvatura.
    
    Clasificación:
    - "very_negative": curvature < negative_threshold (candidatas a eliminar)
    - "negative": negative_threshold <= curvature < 0
    - "zero": curvature == 0
    - "positive": 0 < curvature <= positive_threshold
    - "very_positive": curvature > positive_threshold (conexiones fuertes)
    
    Args:
        curvatures: Diccionario de curvaturas
        negative_threshold: Umbral para clasificar como "muy negativo"
        positive_threshold: Umbral para clasificar como "muy positivo"
    
    Returns:
        Diccionario con listas de aristas por categoría
    """
    classified = {
        "very_negative": [],
        "negative": [],
        "zero": [],
        "positive": [],
        "very_positive": [],
    }
    
    for edge, curv in curvatures.items():
        if curv < negative_threshold:
            classified["very_negative"].append(edge)
        elif curv < 0:
            classified["negative"].append(edge)
        elif curv == 0:
            classified["zero"].append(edge)
        elif curv <= positive_threshold:
            classified["positive"].append(edge)
        else:
            classified["very_positive"].append(edge)
    
    return classified


# Alias para consistencia con GraphRicciCurvature API
compute_ricci_curvature = compute_forman_ricci

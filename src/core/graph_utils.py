"""
Generadores de grafos sintéticos para benchmarks.

Estos grafos sirven como escenarios de prueba para evaluar:
- Ricci-cleaning (Fase 1)
- Multi-espacio MoE (Fase 2)
- Routing y métricas (Fase 3)

VARIABLES AJUSTABLES (marcadas con # PARAM):
- Todos los valores por defecto son experimentables
- Ver README.md para rangos recomendados
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from typing import Literal


def generate_tree(
    branching: int = 3,  # PARAM: factor de ramificación
    depth: int = 5,  # PARAM: profundidad del árbol
    seed: int | None = None,
) -> nx.Graph:
    """
    Genera un árbol jerárquico perfecto.
    
    Útil para probar embeddings hiperbólicos (H los captura bien).
    
    Args:
        branching: Número de hijos por nodo
        depth: Profundidad del árbol
        seed: Semilla para reproducibilidad (no afecta estructura, solo IDs)
    
    Returns:
        Grafo árbol con (branching^depth - 1) / (branching - 1) nodos
    """
    # balanced_tree genera árbol perfecto
    G = nx.balanced_tree(r=branching, h=depth)
    
    # Añadir atributo de nivel para análisis
    for node in G.nodes():
        # Calcular nivel basado en posición
        level = 0
        temp = node
        while temp > 0:
            temp = (temp - 1) // branching
            level += 1
        G.nodes[node]["level"] = depth - level
    
    return G


def generate_sbm_graph(
    sizes: list[int] | None = None,  # PARAM: tamaños de comunidades
    p_in: float = 0.3,  # PARAM: prob. conexión intra-comunidad
    p_out: float = 0.05,  # PARAM: prob. conexión inter-comunidad
    seed: int | None = None,
) -> nx.Graph:
    """
    Genera un grafo Stochastic Block Model (comunidades).
    
    Útil para probar detección de estructura y routing entre clusters.
    
    Args:
        sizes: Lista con tamaño de cada comunidad (default: [50, 50, 50])
        p_in: Probabilidad de conexión dentro de comunidad
        p_out: Probabilidad de conexión entre comunidades
        seed: Semilla para reproducibilidad
    
    Returns:
        Grafo con comunidades marcadas en atributo 'community'
    """
    if sizes is None:
        sizes = [50, 50, 50]  # PARAM: default 3 comunidades de 50
    
    n_communities = len(sizes)
    
    # Matriz de probabilidades: p_in en diagonal, p_out fuera
    p_matrix = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(p_matrix, p_in)
    
    G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    
    # El SBM ya añade 'block' como atributo, renombramos a 'community'
    for node in G.nodes():
        G.nodes[node]["community"] = G.nodes[node].get("block", 0)
    
    return G


def generate_grid_with_noise(
    rows: int = 20,  # PARAM: filas del grid
    cols: int = 20,  # PARAM: columnas del grid
    p_rewire: float = 0.1,  # PARAM: prob. de rewiring por arista
    seed: int | None = None,
) -> nx.Graph:
    """
    Genera un grid 2D con rewiring aleatorio (small-world style).
    
    Útil para probar efectos de atajos y estructura local.
    
    Args:
        rows: Número de filas
        cols: Número de columnas
        p_rewire: Probabilidad de reconectar cada arista a nodo aleatorio
        seed: Semilla para reproducibilidad
    
    Returns:
        Grafo grid con algunas aristas reconectadas
    """
    rng = np.random.default_rng(seed)
    
    # Crear grid base
    G = nx.grid_2d_graph(rows, cols)
    
    # Convertir nodos de tupla a enteros para consistencia
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Añadir coordenadas originales como atributos
    inverse_mapping = {v: k for k, v in mapping.items()}
    for node in G.nodes():
        row, col = inverse_mapping[node]
        G.nodes[node]["row"] = row
        G.nodes[node]["col"] = col
    
    # Rewiring aleatorio
    edges_to_remove = []
    edges_to_add = []
    nodes = list(G.nodes())
    
    for u, v in list(G.edges()):
        if rng.random() < p_rewire:
            # Elegir nuevo destino aleatorio (evitando self-loops y duplicados)
            new_target = rng.choice(nodes)
            attempts = 0
            while (new_target == u or G.has_edge(u, new_target)) and attempts < 10:
                new_target = rng.choice(nodes)
                attempts += 1
            
            if attempts < 10:
                edges_to_remove.append((u, v))
                edges_to_add.append((u, new_target))
    
    G.remove_edges_from(edges_to_remove)
    G.add_edges_from(edges_to_add)
    
    # Marcar aristas rewired
    for u, v in edges_to_add:
        G.edges[u, v]["rewired"] = True
    
    return G


def add_noise_to_graph(
    G: nx.Graph,
    p_add: float = 0.05,  # PARAM: prob. añadir arista
    p_remove: float = 0.05,  # PARAM: prob. remover arista
    seed: int | None = None,
) -> nx.Graph:
    """
    Añade ruido a un grafo existente (añadiendo/removiendo aristas).
    
    Útil para probar robustez de Ricci-cleaning.
    
    Args:
        G: Grafo original
        p_add: Probabilidad de añadir arista entre cada par de nodos no conectados
        p_remove: Probabilidad de remover cada arista existente
        seed: Semilla para reproducibilidad
    
    Returns:
        Copia del grafo con ruido añadido
    """
    rng = np.random.default_rng(seed)
    G_noisy = G.copy()
    
    nodes = list(G_noisy.nodes())
    n = len(nodes)
    
    # Remover aristas
    edges_to_remove = [
        (u, v) for u, v in G_noisy.edges() 
        if rng.random() < p_remove
    ]
    G_noisy.remove_edges_from(edges_to_remove)
    
    # Añadir aristas (sampling para eficiencia)
    # Número esperado de aristas a añadir
    max_possible_edges = n * (n - 1) // 2 - G_noisy.number_of_edges()
    expected_new_edges = int(max_possible_edges * p_add)
    
    if expected_new_edges > 0:
        # Samplear pares aleatorios
        for _ in range(expected_new_edges):
            u, v = rng.choice(nodes, size=2, replace=False)
            if not G_noisy.has_edge(u, v):
                G_noisy.add_edge(u, v, noisy=True)
    
    return G_noisy


# Tipo para seleccionar generador
GraphType = Literal["tree", "sbm", "grid"]


def generate_graph(
    graph_type: GraphType,
    seed: int | None = None,
    **kwargs,
) -> nx.Graph:
    """
    Factory function para generar grafos por tipo.
    
    Args:
        graph_type: Tipo de grafo ("tree", "sbm", "grid")
        seed: Semilla para reproducibilidad
        **kwargs: Argumentos específicos del generador
    
    Returns:
        Grafo generado
    """
    generators = {
        "tree": generate_tree,
        "sbm": generate_sbm_graph,
        "grid": generate_grid_with_noise,
    }
    
    if graph_type not in generators:
        raise ValueError(f"Tipo de grafo desconocido: {graph_type}. Opciones: {list(generators.keys())}")
    
    return generators[graph_type](seed=seed, **kwargs)

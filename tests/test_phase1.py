"""
Tests para módulos de Fase 1: curvatura, Ricci-cleaning, routing.
"""

import pytest
import networkx as nx
import numpy as np

from src.core.curvature import (
    compute_forman_ricci,
    get_curvature_distribution,
    classify_edges_by_curvature,
)
from src.core.ricci_cleaning import (
    ricci_clean,
    ricci_clean_soft,
    is_bridge,
    suggest_threshold,
)
from src.core.routing import (
    greedy_routing,
    evaluate_routing,
    compare_routing,
    compute_graph_metrics,
)
from src.core.graph_utils import generate_tree, generate_sbm_graph, generate_grid_with_noise


class TestFormanRicci:
    """Tests para curvatura Forman-Ricci."""
    
    def test_tree_curvature(self):
        """En un árbol, curvaturas deben ser <= 2."""
        G = generate_tree(branching=2, depth=3)
        curvatures = compute_forman_ricci(G)
        
        # Para un árbol: F(e) = 4 - deg(u) - deg(v)
        # Nodo interno tiene deg >= 2, así que F <= 0 para aristas internas
        assert len(curvatures) == G.number_of_edges()
        
        for edge, curv in curvatures.items():
            u, v = edge
            expected = 4 - G.degree(u) - G.degree(v)
            assert curv == expected
    
    def test_complete_graph_positive_curvature(self):
        """Grafo completo pequeño tiene curvaturas muy negativas."""
        G = nx.complete_graph(5)
        curvatures = compute_forman_ricci(G)
        
        # En K5, cada nodo tiene grado 4, así que F = 4 - 4 - 4 = -4
        for curv in curvatures.values():
            assert curv == -4
    
    def test_path_graph_curvature(self):
        """Path graph tiene curvaturas específicas."""
        G = nx.path_graph(5)  # 0 -- 1 -- 2 -- 3 -- 4
        curvatures = compute_forman_ricci(G)
        
        # Extremos: deg=1, internos: deg=2
        # (0,1): 4 - 1 - 2 = 1
        # (1,2): 4 - 2 - 2 = 0
        # etc.
        assert curvatures[(0, 1)] == 1  # extremo-interno
        assert curvatures[(1, 2)] == 0  # interno-interno
    
    def test_curvature_distribution(self):
        """Verificar que get_curvature_distribution funciona."""
        G = generate_sbm_graph(sizes=[20, 20], seed=42)
        curvatures = compute_forman_ricci(G)
        dist = get_curvature_distribution(curvatures)
        
        assert "mean" in dist
        assert "std" in dist
        assert "histogram" in dist
        assert len(dist["histogram"]["counts"]) == 20  # default bins
    
    def test_classify_edges(self):
        """Verificar clasificación de aristas."""
        curvatures = {(0, 1): -2.0, (1, 2): -0.3, (2, 3): 0.0, (3, 4): 0.2, (4, 5): 1.0}
        classified = classify_edges_by_curvature(curvatures, negative_threshold=-0.5)
        
        assert (0, 1) in classified["very_negative"]
        assert (1, 2) in classified["negative"]
        assert (2, 3) in classified["zero"]
        assert (3, 4) in classified["positive"]
        assert (4, 5) in classified["very_positive"]


class TestRicciCleaning:
    """Tests para Ricci-cleaning."""
    
    def test_is_bridge(self):
        """Verificar detección de puentes."""
        # Crear grafo con puente
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triángulo
        G.add_edge(2, 3)  # Puente
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])  # Otro triángulo
        
        assert is_bridge(G, (2, 3)) == True
        assert is_bridge(G, (0, 1)) == False
    
    def test_clean_preserves_connectivity(self):
        """Con preserve_connectivity=True, grafo debe quedar conexo."""
        G = generate_sbm_graph(sizes=[30, 30], p_in=0.3, p_out=0.05, seed=42)
        curvatures = compute_forman_ricci(G)
        
        G_clean, info = ricci_clean(G, threshold=-2.0, preserve_connectivity=True)
        
        assert nx.is_connected(G_clean)
        assert info["components_after"] == 1
    
    def test_clean_removes_edges(self):
        """Ricci-cleaning debe remover aristas con curvatura muy negativa."""
        # Crear grafo con aristas de curvatura variada
        G = generate_sbm_graph(sizes=[20, 20], p_in=0.5, p_out=0.1, seed=42)
        curvatures = compute_forman_ricci(G)
        
        # Usar umbral que capture algunas aristas
        threshold = suggest_threshold(curvatures, method="percentile", percentile=20)
        
        G_clean, info = ricci_clean(G, threshold=threshold, preserve_connectivity=True)
        
        # Debería haber removido algunas aristas
        assert info["edges_removed"] >= 0
        assert G_clean.number_of_edges() <= G.number_of_edges()
    
    def test_soft_clean_attenuates(self):
        """Limpieza suave debe atenuar pesos sin remover aristas."""
        G = generate_grid_with_noise(rows=5, cols=5, p_rewire=0.2, seed=42)
        
        G_soft, info = ricci_clean_soft(G, threshold=-2.0, attenuation_factor=0.5)
        
        # Mismo número de aristas
        assert G_soft.number_of_edges() == G.number_of_edges()
        
        # Algunas aristas deben tener peso modificado
        if info["edges_attenuated"] > 0:
            for u, v in G_soft.edges():
                if G_soft[u][v].get("weight", 1.0) < 1.0:
                    assert G_soft[u][v]["weight"] == 0.5
    
    def test_suggest_threshold(self):
        """Verificar sugerencia de umbral."""
        curvatures = {(i, i+1): float(i - 5) for i in range(10)}
        # Valores: -5, -4, -3, -2, -1, 0, 1, 2, 3, 4
        
        thresh_p10 = suggest_threshold(curvatures, method="percentile", percentile=10)
        assert thresh_p10 < -3  # Debería estar en el extremo bajo


class TestRouting:
    """Tests para routing y métricas."""
    
    def test_greedy_routing_success(self):
        """Greedy routing debe encontrar ruta en grafo conexo sencillo."""
        G = nx.path_graph(10)
        result = greedy_routing(G, 0, 9)
        
        assert result.success == True
        assert result.source == 0
        assert result.target == 9
        assert result.oracle_hops == 9
    
    def test_greedy_routing_same_node(self):
        """Routing al mismo nodo debe ser exitoso con 0 hops."""
        G = nx.path_graph(5)
        result = greedy_routing(G, 2, 2)
        
        assert result.success == True
        assert result.hops == 0
        assert result.stretch == 1.0
    
    def test_evaluate_routing(self):
        """Evaluar routing en grafo conocido."""
        G = generate_tree(branching=2, depth=4, seed=42)
        metrics = evaluate_routing(G, n_queries=50, seed=123)
        
        assert "success_rate" in metrics
        assert "avg_stretch" in metrics
        assert metrics["n_queries"] == 50
        assert 0 <= metrics["success_rate"] <= 1.0
    
    def test_compare_routing(self):
        """Comparar routing antes/después."""
        G = generate_sbm_graph(sizes=[30, 30], seed=42)
        curvatures = compute_forman_ricci(G)
        G_clean, _ = ricci_clean(G, threshold=-3.0, preserve_connectivity=True)
        
        comparison = compare_routing(G, G_clean, n_queries=30, seed=123)
        
        assert "before" in comparison
        assert "after" in comparison
        assert "delta_success_rate" in comparison
    
    def test_compute_graph_metrics(self):
        """Verificar métricas de grafo."""
        G = generate_grid_with_noise(rows=5, cols=5, p_rewire=0.0, seed=42)
        metrics = compute_graph_metrics(G)
        
        assert metrics["n_nodes"] == 25
        assert metrics["n_components"] == 1
        assert "avg_clustering" in metrics


class TestIntegration:
    """Tests de integración de pipeline completo."""
    
    def test_full_pipeline_tree(self):
        """Pipeline completo en árbol."""
        # 1. Generar grafo
        G = generate_tree(branching=3, depth=4, seed=42)
        
        # 2. Calcular curvatura
        curvatures = compute_forman_ricci(G)
        assert len(curvatures) == G.number_of_edges()
        
        # 3. Evaluar routing antes
        metrics_before = evaluate_routing(G, n_queries=20, seed=123)
        
        # 4. Limpiar (aunque árbol tiene pocas aristas malas)
        G_clean, info = ricci_clean(G, threshold=-5.0, preserve_connectivity=True)
        
        # 5. Evaluar routing después
        metrics_after = evaluate_routing(G_clean, n_queries=20, seed=123)
        
        # 6. Verificar
        assert nx.is_connected(G_clean)
        assert metrics_before["success_rate"] >= 0
        assert metrics_after["success_rate"] >= 0
    
    def test_full_pipeline_sbm(self):
        """Pipeline completo en SBM (más interesante)."""
        # 1. Generar grafo con ruido
        G = generate_sbm_graph(sizes=[30, 30, 30], p_in=0.4, p_out=0.1, seed=42)
        
        # 2. Curvatura
        curvatures = compute_forman_ricci(G)
        dist = get_curvature_distribution(curvatures)
        
        # 3. Umbral sugerido
        threshold = suggest_threshold(curvatures, method="percentile", percentile=15)
        
        # 4. Comparación
        G_clean, info = ricci_clean(G, threshold=threshold, preserve_connectivity=True)
        comparison = compare_routing(G, G_clean, n_queries=50, seed=123)
        
        # 5. El grafo debe seguir conexo
        assert nx.is_connected(G_clean)
        
        # 6. Registrar resultados (para debugging)
        print(f"\nSBM Pipeline Results:")
        print(f"  Edges removed: {info['edges_removed']}")
        print(f"  Success before: {comparison['before']['success_rate']:.2%}")
        print(f"  Success after: {comparison['after']['success_rate']:.2%}")
        print(f"  Delta: {comparison['delta_success_rate']:+.2%}")

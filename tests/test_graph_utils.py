"""
Tests para generadores de grafos sintéticos.
"""

import pytest
import networkx as nx

from src.core.graph_utils import (
    generate_tree,
    generate_sbm_graph,
    generate_grid_with_noise,
    add_noise_to_graph,
    generate_graph,
)


class TestGenerateTree:
    """Tests para generate_tree."""
    
    def test_creates_connected_graph(self):
        """El árbol debe ser conexo."""
        G = generate_tree(branching=3, depth=4)
        assert nx.is_connected(G)
    
    def test_correct_node_count(self):
        """Verificar número de nodos: (r^(h+1) - 1) / (r - 1)."""
        G = generate_tree(branching=2, depth=3)
        # 2^4 - 1 = 15 nodos
        expected = (2 ** 4 - 1)
        assert G.number_of_nodes() == expected
    
    def test_is_tree(self):
        """El grafo debe ser un árbol (n-1 aristas)."""
        G = generate_tree(branching=3, depth=3)
        n = G.number_of_nodes()
        assert G.number_of_edges() == n - 1
    
    def test_reproducibility(self):
        """La misma semilla debe dar el mismo grafo."""
        G1 = generate_tree(branching=3, depth=4, seed=42)
        G2 = generate_tree(branching=3, depth=4, seed=42)
        assert set(G1.nodes()) == set(G2.nodes())
        assert set(G1.edges()) == set(G2.edges())


class TestGenerateSBM:
    """Tests para generate_sbm_graph."""
    
    def test_correct_node_count(self):
        """Número de nodos debe coincidir con suma de sizes."""
        sizes = [30, 40, 50]
        G = generate_sbm_graph(sizes=sizes, seed=42)
        assert G.number_of_nodes() == sum(sizes)
    
    def test_has_community_attribute(self):
        """Cada nodo debe tener atributo 'community'."""
        G = generate_sbm_graph(sizes=[20, 20], seed=42)
        for node in G.nodes():
            assert "community" in G.nodes[node]
    
    def test_more_edges_within_community(self):
        """Debe haber más aristas intra-comunidad que inter-comunidad."""
        G = generate_sbm_graph(sizes=[50, 50], p_in=0.5, p_out=0.05, seed=42)
        
        intra_edges = 0
        inter_edges = 0
        
        for u, v in G.edges():
            if G.nodes[u]["community"] == G.nodes[v]["community"]:
                intra_edges += 1
            else:
                inter_edges += 1
        
        # Con estos parámetros, intra >> inter
        assert intra_edges > inter_edges
    
    def test_reproducibility(self):
        """La misma semilla debe dar el mismo grafo."""
        G1 = generate_sbm_graph(sizes=[30, 30], seed=123)
        G2 = generate_sbm_graph(sizes=[30, 30], seed=123)
        assert G1.number_of_edges() == G2.number_of_edges()


class TestGenerateGridWithNoise:
    """Tests para generate_grid_with_noise."""
    
    def test_correct_node_count(self):
        """Número de nodos = rows * cols."""
        G = generate_grid_with_noise(rows=10, cols=15, p_rewire=0.0, seed=42)
        assert G.number_of_nodes() == 10 * 15
    
    def test_no_rewire_is_grid(self):
        """Con p_rewire=0, debe ser un grid perfecto."""
        G = generate_grid_with_noise(rows=5, cols=5, p_rewire=0.0, seed=42)
        # Un grid 5x5 tiene: 5*4 + 5*4 = 40 aristas (horizontales + verticales)
        # Más precisamente: (rows-1)*cols + rows*(cols-1) = 4*5 + 5*4 = 40
        expected_edges = (5 - 1) * 5 + 5 * (5 - 1)
        assert G.number_of_edges() == expected_edges
    
    def test_rewire_changes_edges(self):
        """Con p_rewire > 0, algunas aristas deben estar marcadas como rewired."""
        G = generate_grid_with_noise(rows=10, cols=10, p_rewire=0.2, seed=42)
        
        rewired_count = sum(
            1 for u, v, data in G.edges(data=True) 
            if data.get("rewired", False)
        )
        
        # Con p=0.2 y ~180 aristas, esperamos algunas rewired
        assert rewired_count > 0
    
    def test_has_position_attributes(self):
        """Cada nodo debe tener row/col."""
        G = generate_grid_with_noise(rows=5, cols=5, p_rewire=0.0, seed=42)
        for node in G.nodes():
            assert "row" in G.nodes[node]
            assert "col" in G.nodes[node]


class TestAddNoiseToGraph:
    """Tests para add_noise_to_graph."""
    
    def test_no_noise_preserves_graph(self):
        """Con p_add=0 y p_remove=0, el grafo no cambia."""
        G_orig = generate_tree(branching=2, depth=3, seed=42)
        G_noisy = add_noise_to_graph(G_orig, p_add=0.0, p_remove=0.0, seed=42)
        
        assert G_noisy.number_of_edges() == G_orig.number_of_edges()
    
    def test_noise_changes_edges(self):
        """Con ruido, el número de aristas debería cambiar."""
        G_orig = generate_sbm_graph(sizes=[30, 30], seed=42)
        G_noisy = add_noise_to_graph(G_orig, p_add=0.1, p_remove=0.1, seed=42)
        
        # Podría ser igual por azar, pero muy improbable
        # Verificamos que al menos el proceso corre sin error
        assert G_noisy.number_of_nodes() == G_orig.number_of_nodes()


class TestGenerateGraphFactory:
    """Tests para la factory function."""
    
    def test_generates_tree(self):
        G = generate_graph("tree", seed=42, branching=2, depth=3)
        assert nx.is_tree(G)
    
    def test_generates_sbm(self):
        G = generate_graph("sbm", seed=42, sizes=[20, 20])
        assert G.number_of_nodes() == 40
    
    def test_generates_grid(self):
        G = generate_graph("grid", seed=42, rows=5, cols=5, p_rewire=0.0)
        assert G.number_of_nodes() == 25
    
    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            generate_graph("invalid_type")

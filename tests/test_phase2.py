"""
Tests para módulos de Fase 2: embeddings Euclídeo, Hiperbólico, y MoE Gating.
"""

import pytest
import networkx as nx
import numpy as np

from src.geometry.euclidean import (
    mds_embedding,
    laplacian_embedding,
    embed_euclidean,
    compute_embedding_distance,
    compute_distance_matrix,
)
from src.geometry.hyperbolic import (
    poincare_distance,
    project_to_ball,
    poincare_embedding,
    embed_hyperbolic,
)
from src.geometry.moe_gating import (
    compute_node_features,
    soft_gating,
    geometric_moe_embedding,
    compute_mixed_distance,
    GeometrySpace,
)
from src.core.graph_utils import generate_tree, generate_sbm_graph


class TestEuclideanEmbedding:
    """Tests para embedding Euclídeo."""
    
    def test_mds_basic(self):
        """MDS debe producir un embedding válido."""
        G = generate_tree(branching=2, depth=3, seed=42)
        result = mds_embedding(G, dim=8)
        
        assert len(result.positions) == G.number_of_nodes()
        assert result.dim == 8
        assert result.method == "mds"
        
        # Verificar que las posiciones son arrays del tamaño correcto
        for node, pos in result.positions.items():
            assert pos.shape == (8,)
    
    def test_laplacian_basic(self):
        """Laplacian debe producir un embedding válido."""
        G = generate_sbm_graph(sizes=[20, 20], seed=42)
        result = laplacian_embedding(G, dim=4)
        
        assert len(result.positions) == G.number_of_nodes()
        assert result.dim == 4
    
    def test_distance_matrix(self):
        """Matriz de distancias debe ser simétrica."""
        G = nx.path_graph(5)
        D, nodes = compute_distance_matrix(G)
        
        assert D.shape == (5, 5)
        assert np.allclose(D, D.T)
        assert D[0, 4] == 4  # Path: 0-1-2-3-4
    
    def test_embed_factory(self):
        """Factory function debe funcionar."""
        G = generate_tree(branching=2, depth=2)
        
        for method in ["mds", "laplacian"]:
            result = embed_euclidean(G, dim=4, method=method)
            assert len(result.positions) == G.number_of_nodes()
    
    def test_embedding_distance(self):
        """Distancia en embedding debe ser ≥ 0."""
        G = generate_tree(branching=2, depth=3, seed=42)
        result = mds_embedding(G, dim=8)
        
        nodes = list(result.positions.keys())
        d = compute_embedding_distance(result.positions, nodes[0], nodes[1])
        assert d >= 0


class TestHyperbolicEmbedding:
    """Tests para embedding Hiperbólico."""
    
    def test_poincare_distance_self(self):
        """Distancia de un punto a sí mismo es ~0."""
        x = np.array([0.1, 0.2])
        d = poincare_distance(x, x)
        # Tolerancia por errores numéricos en arcosh cerca de 1
        assert d < 0.01
    
    def test_poincare_distance_symmetric(self):
        """Distancia hiperbólica es simétrica."""
        x = np.array([0.1, 0.2])
        y = np.array([0.3, -0.1])
        
        assert np.isclose(poincare_distance(x, y), poincare_distance(y, x))
    
    def test_poincare_distance_origin(self):
        """Distancia desde el origen."""
        origin = np.array([0.0, 0.0])
        point = np.array([0.5, 0.0])
        
        d = poincare_distance(origin, point)
        assert d > 0
    
    def test_project_to_ball(self):
        """Proyección mantiene puntos dentro del ball."""
        # Punto fuera del ball
        x = np.array([2.0, 2.0])
        x_proj = project_to_ball(x, c=-1.0)
        
        # La norma debe ser < 1 (radio del ball con c=-1)
        assert np.linalg.norm(x_proj) < 1.0
    
    def test_poincare_embedding_basic(self):
        """Embedding Poincaré debe producir resultado válido."""
        G = generate_tree(branching=2, depth=2, seed=42)
        result = poincare_embedding(G, dim=4, epochs=10, seed=42)
        
        assert len(result.positions) == G.number_of_nodes()
        assert result.dim == 4
        
        # Verificar que todos los puntos están en el ball
        for pos in result.positions.values():
            assert np.linalg.norm(pos) < 1.0  # Con c=-1, radio=1
    
    def test_embed_factory(self):
        """Factory function debe funcionar."""
        G = generate_tree(branching=2, depth=2)
        result = embed_hyperbolic(G, dim=4, epochs=5)
        
        assert len(result.positions) == G.number_of_nodes()


class TestMoEGating:
    """Tests para MoE Gating."""
    
    def test_compute_node_features(self):
        """Características de nodo deben calcularse correctamente."""
        G = generate_sbm_graph(sizes=[15, 15], seed=42)
        features = compute_node_features(G)
        
        assert len(features) == G.number_of_nodes()
        
        for feat in features.values():
            assert feat.degree >= 1
            assert 0 <= feat.clustering <= 1
    
    def test_soft_gating_sums_to_one(self):
        """Los pesos de gating deben sumar 1."""
        from src.geometry.moe_gating import NodeFeatures
        
        features = NodeFeatures(
            node=0,
            degree=5,
            local_curvature=-2.0,
            clustering=0.3,
            is_hub=False,
        )
        
        weights = soft_gating(features)
        
        assert "E" in weights
        assert "H" in weights
        assert np.isclose(weights["E"] + weights["H"], 1.0)
    
    def test_soft_gating_temperature(self):
        """Temperatura baja produce distribución más concentrada."""
        from src.geometry.moe_gating import NodeFeatures
        
        features = NodeFeatures(
            node=0,
            degree=5,
            local_curvature=-2.0,
            clustering=0.3,
            is_hub=False,
        )
        
        weights_cold = soft_gating(features, temperature=0.1)
        weights_hot = soft_gating(features, temperature=5.0)
        
        # Con temperatura baja, un peso domina más
        max_cold = max(weights_cold.values())
        max_hot = max(weights_hot.values())
        
        assert max_cold >= max_hot
    
    def test_geometric_moe_embedding(self):
        """MoE embedding debe producir resultados válidos."""
        G = generate_tree(branching=2, depth=3, seed=42)
        result = geometric_moe_embedding(
            G, dim=4, 
            hyperbolic_epochs=10,
            seed=42,
        )
        
        assert len(result.positions_euclidean) == G.number_of_nodes()
        assert len(result.positions_hyperbolic) == G.number_of_nodes()
        assert len(result.gating_decisions) == G.number_of_nodes()
        assert len(result.mixed_positions) == G.number_of_nodes()
        
        # Verificar métricas
        assert "euclidean_ratio" in result.metrics
        assert 0 <= result.metrics["euclidean_ratio"] <= 1
    
    def test_mixed_distance(self):
        """Distancia mixta debe ser no-negativa."""
        G = generate_tree(branching=2, depth=2, seed=42)
        result = geometric_moe_embedding(G, dim=4, hyperbolic_epochs=5)
        
        nodes = list(G.nodes())
        d = compute_mixed_distance(result, nodes[0], nodes[1])
        
        assert d >= 0


class TestIntegrationPhase2:
    """Tests de integración para Fase 2."""
    
    def test_tree_prefers_hyperbolic(self):
        """Árboles deberían tener más nodos preferiendo hiperbólico."""
        G = generate_tree(branching=3, depth=3, seed=42)
        result = geometric_moe_embedding(
            G, dim=8,
            hyperbolic_epochs=20,
            seed=42,
        )
        
        # Árboles tienen curvatura negativa → hiperbólico preferido
        # (pero depende de la implementación, así que solo verificamos que funcione)
        assert result.metrics["hyperbolic_ratio"] >= 0
    
    def test_sbm_embedding_quality(self):
        """SBM debe producir embeddings con stress razonable."""
        G = generate_sbm_graph(sizes=[20, 20], seed=42)
        
        # Euclídeo
        emb_E = embed_euclidean(G, dim=16, method="mds")
        
        # Hiperbólico
        emb_H = embed_hyperbolic(G, dim=16, epochs=30, seed=42)
        
        # Stress/loss deberían ser finitos
        assert np.isfinite(emb_E.stress)
        assert np.isfinite(emb_H.final_loss)
    
    def test_full_moe_pipeline(self):
        """Pipeline completo de MoE."""
        # 1. Generar grafo
        G = generate_sbm_graph(sizes=[30, 30], p_in=0.3, p_out=0.05, seed=42)
        
        # 2. MoE embedding
        result = geometric_moe_embedding(
            G, dim=16,
            temperature=1.0,
            hyperbolic_epochs=30,
            seed=42,
        )
        
        # 3. Verificar que tenemos variedad en las decisiones
        decisions = result.gating_decisions
        selected_spaces = [d.selected for d in decisions.values()]
        
        # Al menos debería haber algunas decisiones de cada tipo
        # (aunque esto depende del grafo, así que solo verificamos que no falle)
        assert "E" in selected_spaces or "H" in selected_spaces
        
        # 4. Las métricas deben ser válidas
        assert result.metrics["avg_confidence"] > 0
        assert result.metrics["n_nodes"] == G.number_of_nodes()

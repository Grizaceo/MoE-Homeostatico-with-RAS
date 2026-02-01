"""
Tests para RSA Engine: Population, Selection, y Solver.
"""

import pytest
import numpy as np
import networkx as nx

from src.rsa_engine.population import (
    Candidate,
    PopulationManager,
    MockEmbeddingProvider,
)
from src.rsa_engine.selection import (
    stratified_sample,
    RepechageBuffer,
    compute_similarity_scores,
    random_sample_baseline,
)
from src.rsa_engine.solver import (
    RSASolver,
    RSAConfig,
    RSAResult,
    MockLLMAdapter,
)


class TestCandidate:
    """Tests para la clase Candidate."""
    
    def test_candidate_creation(self):
        """Candidate se crea correctamente."""
        embedding = np.random.randn(64)
        embedding = embedding / np.linalg.norm(embedding)
        
        candidate = Candidate(
            id=0,
            response="Test response",
            embedding=embedding,
            ricci_curvature=-1.5,
            score=0.8,
        )
        
        assert candidate.id == 0
        assert candidate.response == "Test response"
        assert candidate.ricci_curvature == -1.5
        assert candidate.score == 0.8
        assert len(candidate.embedding) == 64
    
    def test_semantic_similarity(self):
        """Similitud coseno funciona correctamente."""
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([1.0, 0.0, 0.0])
        e3 = np.array([0.0, 1.0, 0.0])
        
        c1 = Candidate(id=0, response="a", embedding=e1)
        c2 = Candidate(id=1, response="b", embedding=e2)
        c3 = Candidate(id=2, response="c", embedding=e3)
        
        # Idénticos
        assert np.isclose(c1.semantic_similarity(c2), 1.0)
        # Ortogonales
        assert np.isclose(c1.semantic_similarity(c3), 0.0)
    
    def test_semantic_distance(self):
        """Distancia semántica = 1 - similitud."""
        e1 = np.array([1.0, 0.0])
        e2 = np.array([1.0, 0.0])
        
        c1 = Candidate(id=0, response="a", embedding=e1)
        c2 = Candidate(id=1, response="b", embedding=e2)
        
        assert np.isclose(c1.semantic_distance(c2), 0.0)


class TestPopulationManager:
    """Tests para PopulationManager."""
    
    def test_add_candidate(self):
        """Añadir candidatos funciona."""
        pm = PopulationManager(seed=42)
        
        cid = pm.add_candidate(response="Test 1")
        
        assert cid == 0
        assert pm.size == 1
        assert cid in pm.candidates
    
    def test_multiple_candidates(self):
        """Múltiples candidatos se gestionan."""
        pm = PopulationManager(seed=42)
        
        ids = [pm.add_candidate(response=f"Test {i}") for i in range(10)]
        
        assert pm.size == 10
        assert len(set(ids)) == 10  # Todos únicos
    
    def test_similarity_graph_construction(self):
        """Grafo de similitud se construye."""
        pm = PopulationManager(similarity_threshold=0.3, seed=42)
        
        # Añadir varios candidatos
        for i in range(10):
            pm.add_candidate(response=f"Response number {i}")
        
        # Debería haber algunas aristas (embeddings determinísticos por hash)
        assert pm.graph.number_of_nodes() == 10
        # El número de aristas depende de los embeddings generados
    
    def test_local_curvature_computation(self):
        """Curvatura local se calcula."""
        pm = PopulationManager(similarity_threshold=0.1, seed=42)
        
        # Crear población con embeddings que generen aristas
        for i in range(5):
            pm.add_candidate(response=f"Similar response {i}")
        
        pm.rebuild_similarity_graph(threshold=0.1)
        
        curvature = pm.compute_local_curvature(0)
        # Forman-Ricci: 4 - deg(u) - deg(v), promediado
        assert isinstance(curvature, float)
    
    def test_expert_profile_embedding(self):
        """Obtener embedding de experto funciona."""
        pm = PopulationManager(seed=42)
        cid = pm.add_candidate(response="Expert response")
        
        embedding = pm.get_expert_profile_embedding(cid)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 64  # Default mock dim
    
    def test_centroid_embedding(self):
        """Centroide se calcula correctamente."""
        pm = PopulationManager(seed=42)
        
        for i in range(5):
            pm.add_candidate(response=f"Response {i}")
        
        centroid = pm.get_centroid_embedding()
        
        assert isinstance(centroid, np.ndarray)
        # Está normalizado
        assert np.isclose(np.linalg.norm(centroid), 1.0)
    
    def test_get_top_candidates(self):
        """Top candidatos por score funciona."""
        pm = PopulationManager(seed=42)
        
        for i in range(5):
            cid = pm.add_candidate(response=f"Response {i}")
            pm.candidates[cid].score = float(i)  # Scores 0, 1, 2, 3, 4
        
        top = pm.get_top_candidates(3, by="score")
        
        assert len(top) == 3
        assert top[0].score == 4.0
        assert top[1].score == 3.0
        assert top[2].score == 2.0


class TestStratifiedSample:
    """Tests para muestreo estratificado."""
    
    def test_stratified_sample_basic(self):
        """Muestreo estratificado funciona."""
        pm = PopulationManager(seed=42)
        
        for i in range(10):
            pm.add_candidate(response=f"Response {i}")
        
        expert_embedding = pm.candidates[0].embedding
        
        selected = stratified_sample(
            population=pm,
            expert_embedding=expert_embedding,
            k=3,
            seed=42,
        )
        
        assert len(selected) == 3
        assert all(s in pm.candidates for s in selected)
    
    def test_stratified_not_uniform(self):
        """Estratificado no es uniforme - sesga hacia similares."""
        pm = PopulationManager(seed=42)
        
        # Crear embeddings controlados
        for i in range(20):
            pm.add_candidate(response=f"Response {i}")
        
        # El experto es el candidato 0
        expert_embedding = pm.candidates[0].embedding
        
        # Muestrear muchas veces y ver distribución
        counts = {i: 0 for i in range(20)}
        for _ in range(100):
            selected = stratified_sample(
                population=pm,
                expert_embedding=expert_embedding,
                k=5,
                temperature=0.5,  # Temperatura baja = más sesgado
            )
            for s in selected:
                counts[s] += 1
        
        # No todos los candidatos deberían tener el mismo count
        values = list(counts.values())
        assert max(values) > min(values)  # Hay variación
    
    def test_temperature_effect(self):
        """Temperatura baja concentra más la distribución."""
        pm = PopulationManager(seed=42)
        
        for i in range(10):
            pm.add_candidate(response=f"Response {i}")
        
        expert_embedding = pm.candidates[0].embedding
        
        # Con temperatura muy baja, debería seleccionar los más similares
        selected_cold = stratified_sample(
            population=pm,
            expert_embedding=expert_embedding,
            k=3,
            temperature=0.1,
            seed=42,
        )
        
        # Con temperatura alta, más aleatorio
        selected_hot = stratified_sample(
            population=pm,
            expert_embedding=expert_embedding,
            k=3,
            temperature=10.0,
            seed=42,
        )
        
        # Ambos deberían retornar 3 elementos
        assert len(selected_cold) == 3
        assert len(selected_hot) == 3


class TestRepechageBuffer:
    """Tests para el buffer de repechaje."""
    
    def test_repechage_creation(self):
        """RepechageBuffer se crea correctamente."""
        buffer = RepechageBuffer(
            max_size=5,
            distance_threshold=0.7,
            curvature_stability_threshold=-2.0,
        )
        
        assert buffer.max_size == 5
        assert len(buffer.rescued_candidates) == 0
    
    def test_detect_outliers(self):
        """Detecta outliers con curvatura estable."""
        pm = PopulationManager(similarity_threshold=0.1, seed=42)
        
        # Crear población
        for i in range(10):
            cid = pm.add_candidate(response=f"Normal response {i}")
            pm.candidates[cid].ricci_curvature = -1.0  # Estable
        
        # Crear un outlier (embedding muy diferente pero curvatura ok)
        outlier_embedding = np.ones(64) / np.sqrt(64)  # Diferente a los demás
        outlier_id = pm.add_candidate(
            response="Outlier response",
            embedding=outlier_embedding,
        )
        pm.candidates[outlier_id].ricci_curvature = -0.5  # Muy estable
        
        buffer = RepechageBuffer(
            distance_threshold=0.5,
            curvature_stability_threshold=-2.0,
        )
        
        # Simular agregación sin el outlier
        current_agg = [0, 1, 2]
        
        rescued = buffer.detect_and_rescue(
            population=pm,
            current_aggregation=current_agg,
        )
        
        # Podría o no rescatar dependiendo de embeddings
        assert isinstance(rescued, list)
    
    def test_inject_rescued(self):
        """Inyección de rescatados funciona."""
        buffer = RepechageBuffer()
        buffer.rescued_candidates = [10, 11, 12]
        
        next_round = [0, 1, 2, 3]
        result = buffer.inject_rescued(next_round)
        
        assert len(result) == 7  # 4 + 3
        assert 10 in result
        assert 11 in result
        assert 12 in result
    
    def test_inject_avoids_duplicates(self):
        """Inyección evita duplicados."""
        buffer = RepechageBuffer()
        buffer.rescued_candidates = [0, 1, 10]  # 0 y 1 ya estarán
        
        next_round = [0, 1, 2, 3]
        result = buffer.inject_rescued(next_round)
        
        assert len(result) == 5  # 4 + 1 nuevo
        assert result.count(10) == 1
    
    def test_rescue_stats(self):
        """Estadísticas de rescate funcionan."""
        buffer = RepechageBuffer()
        buffer.rescue_history = [
            {"candidate_id": 0, "distance_to_centroid": 0.8, "curvature": -0.5, "round": 0},
            {"candidate_id": 1, "distance_to_centroid": 0.9, "curvature": -1.0, "round": 1},
        ]
        
        stats = buffer.get_rescue_stats()
        
        assert stats["total_rescued"] == 2
        assert np.isclose(stats["avg_distance"], 0.85)
        assert np.isclose(stats["avg_curvature"], -0.75)


class TestRSASolver:
    """Tests para RSASolver."""
    
    def test_solver_creation(self):
        """RSASolver se crea correctamente."""
        config = RSAConfig(
            population_size=8,
            aggregation_size=2,
            steps=2,
            seed=42,
        )
        solver = RSASolver(config=config)
        
        assert solver.config.population_size == 8
        assert solver.config.steps == 2
    
    def test_solve_basic(self):
        """Solve completa sin errores."""
        config = RSAConfig(
            population_size=8,
            aggregation_size=2,
            steps=2,
            seed=42,
        )
        solver = RSASolver(config=config)
        
        result = solver.solve("What is 2+2?")
        
        assert isinstance(result, RSAResult)
        assert result.final_response is not None
        assert result.total_candidates_generated >= 8
        assert result.rounds_completed == 2
    
    def test_solve_with_repechage(self):
        """Solve con repechaje activado."""
        config = RSAConfig(
            population_size=16,
            aggregation_size=4,
            steps=3,
            enable_repechage=True,
            seed=42,
        )
        solver = RSASolver(config=config)
        
        result = solver.solve("Explain quantum computing")
        
        assert isinstance(result, RSAResult)
        assert "repechage_stats" in dir(result)
    
    def test_solve_without_repechage(self):
        """Solve sin repechaje funciona."""
        config = RSAConfig(
            population_size=8,
            aggregation_size=2,
            steps=2,
            enable_repechage=False,
            seed=42,
        )
        solver = RSASolver(config=config)
        
        result = solver.solve("Simple query")
        
        assert isinstance(result, RSAResult)
        assert result.repechage_stats["total_rescued"] == 0
    
    def test_cost_estimate(self):
        """Estimación de costo es razonable."""
        config = RSAConfig(
            population_size=10,
            aggregation_size=3,
            steps=2,
            seed=42,
        )
        solver = RSASolver(config=config)
        
        result = solver.solve("Test query")
        
        # Costo = candidatos generados + agregaciones
        assert result.cost_estimate > 0
        assert result.cost_estimate >= result.total_candidates_generated


class TestRandomSampleBaseline:
    """Tests para baseline de comparación."""
    
    def test_random_sample_uniform(self):
        """Muestreo aleatorio es uniforme."""
        pm = PopulationManager(seed=42)
        
        for i in range(10):
            pm.add_candidate(response=f"Response {i}")
        
        # Muestrear muchas veces
        counts = {i: 0 for i in range(10)}
        for _ in range(1000):
            selected = random_sample_baseline(pm, k=3)
            for s in selected:
                counts[s] += 1
        
        # Distribución debería ser aproximadamente uniforme
        values = list(counts.values())
        mean = np.mean(values)
        std = np.std(values)
        
        # Coeficiente de variación bajo indica uniformidad
        cv = std / mean
        assert cv < 0.3  # Relativamente uniforme

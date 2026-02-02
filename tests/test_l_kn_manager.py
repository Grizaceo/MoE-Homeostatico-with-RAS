"""
Tests para RAS Controller.
"""

import pytest
import numpy as np

from src.l_kn_core.manager import (
    L_kn_Manager,
    QueryContext,
    RSABudget,
    BudgetProfile,
)


class TestRSABudget:
    """Tests para RSABudget."""
    
    def test_budget_creation(self):
        """RSABudget se crea correctamente."""
        budget = RSABudget(
            population_size=16,
            aggregation_size=4,
            steps=3,
        )
        
        assert budget.population_size == 16
        assert budget.aggregation_size == 4
        assert budget.steps == 3
    
    def test_total_cost(self):
        """Costo total se calcula."""
        budget = RSABudget(
            population_size=16,
            aggregation_size=4,
            steps=3,
        )
        
        # N * T
        assert budget.total_cost == 48
    
    def test_efficiency_ratio(self):
        """Ratio de eficiencia K/N."""
        budget = RSABudget(
            population_size=16,
            aggregation_size=4,
            steps=3,
        )
        
        assert budget.efficiency_ratio == 0.25


class TestQueryContext:
    """Tests para QueryContext."""
    
    def test_context_creation(self):
        """QueryContext se crea."""
        context = QueryContext(
            query="What is AI?",
            estimated_complexity=0.5,
        )
        
        assert context.query == "What is AI?"
        assert context.estimated_complexity == 0.5
    
    def test_context_with_embedding(self):
        """Contexto con embedding."""
        embedding = np.random.randn(64)
        context = QueryContext(
            query="Complex query",
            query_embedding=embedding,
            estimated_complexity=0.8,
        )
        
        assert context.query_embedding is not None
        assert len(context.query_embedding) == 64


class TestBudgetProfiles:
    """Tests para perfiles de presupuesto."""
    
    def test_all_profiles_exist(self):
        """Todos los perfiles están definidos."""
        controller = L_kn_Manager()
        
        for profile in BudgetProfile:
            assert profile in controller.BUDGET_PROFILES
    
    def test_profiles_have_valid_values(self):
        """Perfiles tienen valores válidos."""
        controller = L_kn_Manager()
        
        for profile, budget in controller.BUDGET_PROFILES.items():
            assert budget.population_size > 0
            assert budget.aggregation_size > 0
            assert budget.steps > 0
            assert budget.aggregation_size <= budget.population_size
    
    def test_profiles_ordered_by_cost(self):
        """Perfiles ordenados por costo."""
        controller = L_kn_Manager()
        
        costs = [
            controller.BUDGET_PROFILES[BudgetProfile.MINIMAL].total_cost,
            controller.BUDGET_PROFILES[BudgetProfile.LIGHT].total_cost,
            controller.BUDGET_PROFILES[BudgetProfile.STANDARD].total_cost,
            controller.BUDGET_PROFILES[BudgetProfile.INTENSIVE].total_cost,
        ]
        
        assert costs == sorted(costs)


class Test_L_kn_Manager:
    """Tests para L_kn_Manager."""
    
    def test_controller_creation(self):
        """L_kn_Manager se crea."""
        controller = L_kn_Manager(seed=42)
        
        assert controller.default_profile == BudgetProfile.STANDARD
        assert controller.enable_adaptation is True
    
    def test_decide_budget_simple_query(self):
        """Query simple recibe budget bajo."""
        controller = L_kn_Manager(seed=42)
        
        context = QueryContext(
            query="Hi",  # Muy corto
            estimated_complexity=0.1,
        )
        
        budget = controller.decide_budget(context)
        
        assert budget.profile == BudgetProfile.MINIMAL
        assert budget.total_cost < 20
    
    def test_decide_budget_complex_query(self):
        """Query complejo recibe budget alto."""
        controller = L_kn_Manager(seed=42)
        
        context = QueryContext(
            query="Compare and analyze the pros and cons of different approaches to quantum computing, including superconducting qubits, trapped ions, and photonic systems. Explain why each has trade-offs.",
            estimated_complexity=0.9,
        )
        
        budget = controller.decide_budget(context)
        
        assert budget.profile in [BudgetProfile.STANDARD, BudgetProfile.INTENSIVE]
    
    def test_complexity_estimation(self):
        """Estimación de complejidad funciona."""
        controller = L_kn_Manager(seed=42)
        
        # Query simple
        simple_ctx = QueryContext(query="Hello")
        simple_budget = controller.decide_budget(simple_ctx)
        
        # Query complejo
        complex_ctx = QueryContext(
            query="Analyze and compare the implications of different approaches"
        )
        complex_budget = controller.decide_budget(complex_ctx)
        
        # El complejo debería tener mayor costo
        assert complex_budget.total_cost >= simple_budget.total_cost
    
    def test_max_cost_constraint(self):
        """Límite de costo se respeta."""
        controller = L_kn_Manager(seed=42)
        
        context = QueryContext(
            query="Very complex query requiring deep analysis",
            estimated_complexity=0.9,
            max_cost=10,  # Límite muy bajo
        )
        
        budget = controller.decide_budget(context)
        
        assert budget.total_cost <= 10
    
    def test_update_from_outcome(self):
        """Actualización desde resultado funciona."""
        controller = L_kn_Manager(seed=42)
        
        context = QueryContext(query="Test query")
        budget = RSABudget(
            population_size=16,
            aggregation_size=4,
            steps=3,
            profile=BudgetProfile.STANDARD,
        )
        
        # Registrar éxito
        controller.update_from_outcome(
            context=context,
            budget_used=budget,
            final_confidence=0.9,
            success=True,
        )
        
        stats = controller.get_stats()
        assert stats["total_decisions"] == 1
        assert stats["success_rate"] == 1.0
    
    def test_learning_adaptation(self):
        """Adaptación de tasas de éxito."""
        controller = L_kn_Manager(seed=42, adaptation_rate=0.5)
        
        context = QueryContext(query="Test")
        budget = RSABudget(
            population_size=8,
            aggregation_size=3,
            steps=2,
            profile=BudgetProfile.LIGHT,
        )
        
        initial_rate = controller._profile_success_rates[BudgetProfile.LIGHT]
        
        # Registrar fallo
        controller.update_from_outcome(
            context=context,
            budget_used=budget,
            final_confidence=0.3,
            success=False,
        )
        
        new_rate = controller._profile_success_rates[BudgetProfile.LIGHT]
        
        # La tasa debería haber bajado
        assert new_rate < initial_rate
    
    def test_reset_learning(self):
        """Reset de aprendizaje funciona."""
        controller = L_kn_Manager(seed=42)
        
        # Registrar algunos outcomes
        context = QueryContext(query="Test")
        budget = RSABudget(population_size=8, aggregation_size=2, steps=2)
        
        controller.update_from_outcome(context, budget, 0.5, True)
        controller.update_from_outcome(context, budget, 0.5, False)
        
        assert controller.get_stats()["total_decisions"] == 2
        
        controller.reset_learning()
        
        assert controller.get_stats()["total_decisions"] == 0
    
    def test_get_stats(self):
        """Estadísticas retornan correctamente."""
        controller = L_kn_Manager(seed=42)
        
        stats = controller.get_stats()
        
        assert "total_decisions" in stats
        assert "success_rate" in stats
        assert "profile_distribution" in stats


class TestIntegrationRASRSA:
    """Tests de integración RAS + RSA."""
    
    def test_ras_decides_rsa_params(self):
        """RAS decide parámetros que RSA puede usar."""
        from src.rsa_engine.solver import RSASolver, RSAConfig
        
        controller = L_kn_Manager(seed=42)
        
        context = QueryContext(
            query="Explain the theory of relativity",
            estimated_complexity=0.7,
        )
        
        budget = controller.decide_budget(context)
        
        # Usar budget para configurar RSA
        config = RSAConfig(
            population_size=budget.population_size,
            aggregation_size=budget.aggregation_size,
            steps=budget.steps,
            seed=42,
        )
        
        solver = RSASolver(config=config)
        result = solver.solve(context.query)
        
        # El solve debería completar
        assert result.rounds_completed == budget.steps
    
    def test_feedback_loop(self):
        """Loop de feedback RAS -> RSA -> RAS."""
        from src.rsa_engine.solver import RSASolver, RSAConfig
        
        controller = L_kn_Manager(seed=42)
        
        for i in range(3):
            context = QueryContext(query=f"Query number {i}")
            budget = controller.decide_budget(context)
            
            config = RSAConfig(
                population_size=budget.population_size,
                aggregation_size=budget.aggregation_size,
                steps=budget.steps,
                seed=42,
            )
            
            solver = RSASolver(config=config)
            result = solver.solve(context.query)
            
            # Simular evaluación de éxito
            success = result.total_candidates_generated > 5
            
            controller.update_from_outcome(
                context=context,
                budget_used=budget,
                final_confidence=budget.confidence,
                success=success,
            )
        
        stats = controller.get_stats()
        assert stats["total_decisions"] == 3

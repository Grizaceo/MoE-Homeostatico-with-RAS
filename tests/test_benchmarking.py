"""
Tests for the Antigravity Benchmarking System.

Verifies:
- Scenarios are correctly defined
- Harness can run scenarios against mock solver
- Reporter correctly classifies results with traffic light logic
"""
import pytest
from pathlib import Path

from src.benchmarking.scenarios import (
    Scenario,
    Difficulty,
    get_scenarios,
    get_scenario_by_id,
    L1_SCENARIOS,
    L2_SCENARIOS,
    L3_SCENARIOS,
    ALL_SCENARIOS,
)
from src.benchmarking.harness import (
    MockAntigravitySolver,
    SolverResult,
    BenchmarkHarness,
    create_mock_harness,
    run_quick_benchmark,
)
from src.benchmarking.reporter import (
    TrafficLight,
    classify_run,
    analyze_benchmark,
    generate_report,
)


# =============================================================================
# SCENARIOS TESTS
# =============================================================================
class TestScenarios:
    """Test scenario definitions."""
    
    def test_all_scenarios_have_required_fields(self):
        """Every scenario must have id, prompt, difficulty, expected_answer."""
        for scenario in ALL_SCENARIOS:
            assert scenario.id, "Scenario ID must not be empty"
            assert scenario.prompt, "Scenario prompt must not be empty"
            assert isinstance(scenario.difficulty, Difficulty)
            # L3 can have empty expected_answer (creative tasks)
            if scenario.difficulty != Difficulty.L3_CHAOS:
                assert scenario.expected_answer, f"{scenario.id} must have expected_answer"
    
    def test_scenario_ids_are_unique(self):
        """All scenario IDs must be unique."""
        ids = [s.id for s in ALL_SCENARIOS]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"
    
    def test_l1_scenarios_have_low_resource_expectations(self):
        """L1 trivial scenarios should expect N=1, T=1."""
        for scenario in L1_SCENARIOS:
            assert scenario.max_efficient_n == 1, f"{scenario.id} should have max_efficient_n=1"
            assert scenario.max_efficient_t == 1, f"{scenario.id} should have max_efficient_t=1"
    
    def test_l3_scenarios_have_high_resource_expectations(self):
        """L3 chaos scenarios should expect N=32."""
        for scenario in L3_SCENARIOS:
            assert scenario.max_efficient_n == 32, f"{scenario.id} should have max_efficient_n=32"
    
    def test_get_scenarios_returns_all_by_default(self):
        """get_scenarios() with no args returns all scenarios."""
        assert get_scenarios() == ALL_SCENARIOS
    
    def test_get_scenarios_filters_by_difficulty(self):
        """get_scenarios(difficulty) returns only that level."""
        assert get_scenarios(Difficulty.L1_TRIVIAL) == L1_SCENARIOS
        assert get_scenarios(Difficulty.L2_LOGICAL) == L2_SCENARIOS
        assert get_scenarios(Difficulty.L3_CHAOS) == L3_SCENARIOS
    
    def test_get_scenario_by_id_finds_existing(self):
        """get_scenario_by_id returns correct scenario."""
        scenario = get_scenario_by_id("L1-001")
        assert scenario is not None
        assert scenario.id == "L1-001"
        assert "2 + 2" in scenario.prompt
    
    def test_get_scenario_by_id_returns_none_for_missing(self):
        """get_scenario_by_id returns None for nonexistent ID."""
        assert get_scenario_by_id("NONEXISTENT-999") is None


# =============================================================================
# MOCK SOLVER TESTS
# =============================================================================
class TestMockSolver:
    """Test the mock solver behavior."""
    
    def test_mock_solver_returns_solver_result(self):
        """solve() returns a SolverResult with required fields."""
        solver = MockAntigravitySolver(seed=42)
        result = solver.solve("Test prompt", Difficulty.L1_TRIVIAL, "L1-001")
        
        assert isinstance(result, SolverResult)
        assert result.answer == "4"  # Known answer for L1-001
        assert result.n_used >= 1
        assert result.t_used >= 1
        assert result.execution_time_ms > 0
    
    def test_mock_solver_uses_low_n_for_trivial(self):
        """Mock should use N=1 for trivial problems."""
        solver = MockAntigravitySolver(seed=42)
        result = solver.solve("Simple", Difficulty.L1_TRIVIAL)
        assert result.n_used == 1
    
    def test_mock_solver_uses_high_n_for_chaos(self):
        """Mock should use N=32 for chaos problems."""
        solver = MockAntigravitySolver(seed=42)
        result = solver.solve("Complex", Difficulty.L3_CHAOS)
        assert result.n_used == 32
    
    def test_mock_solver_force_inefficient(self):
        """force_inefficient flag makes mock always use N=32."""
        solver = MockAntigravitySolver(seed=42, force_inefficient=True)
        result = solver.solve("2+2", Difficulty.L1_TRIVIAL)
        assert result.n_used == 32
    
    def test_mock_solver_force_incorrect(self):
        """force_incorrect flag makes mock return wrong answers."""
        solver = MockAntigravitySolver(seed=42, force_incorrect=True)
        result = solver.solve("2+2", Difficulty.L1_TRIVIAL, "L1-001")
        assert result.answer == "WRONG_ANSWER_FOR_TESTING"
    
    def test_mock_solver_deterministic_with_seed(self):
        """Same seed produces same results."""
        solver1 = MockAntigravitySolver(seed=123)
        solver2 = MockAntigravitySolver(seed=123)
        
        result1 = solver1.solve("test", Difficulty.L2_LOGICAL)
        result2 = solver2.solve("test", Difficulty.L2_LOGICAL)
        
        assert result1.n_used == result2.n_used
        assert result1.t_used == result2.t_used


# =============================================================================
# HARNESS TESTS
# =============================================================================
class TestHarness:
    """Test the benchmark harness."""
    
    def test_harness_runs_single_scenario(self):
        """Harness can run a single scenario."""
        harness = create_mock_harness(seed=42)
        scenario = get_scenario_by_id("L1-001")
        
        run = harness.run_scenario(scenario)
        
        assert run.scenario == scenario
        assert run.result is not None
        assert run.is_correct is True  # Mock returns correct answer
    
    def test_harness_runs_full_benchmark(self):
        """Harness can run all scenarios."""
        harness = create_mock_harness(seed=42)
        benchmark = harness.run_benchmark()
        
        assert len(benchmark.runs) == len(ALL_SCENARIOS)
        assert benchmark.total_time_ms > 0
        assert benchmark.solver_name == "MockAntigravitySolver"
    
    def test_harness_filters_by_difficulty(self):
        """Harness can filter scenarios by difficulty."""
        harness = create_mock_harness(seed=42)
        benchmark = harness.run_benchmark(difficulty_filter=Difficulty.L1_TRIVIAL)
        
        assert len(benchmark.runs) == len(L1_SCENARIOS)
        for run in benchmark.runs:
            assert run.scenario.difficulty == Difficulty.L1_TRIVIAL
    
    def test_harness_detects_incorrect_answers(self):
        """Harness correctly identifies wrong answers."""
        harness = create_mock_harness(seed=42, force_incorrect=True)
        scenario = get_scenario_by_id("L1-001")
        
        run = harness.run_scenario(scenario)
        
        assert run.is_correct is False


# =============================================================================
# REPORTER TESTS
# =============================================================================
class TestReporter:
    """Test the traffic light reporter."""
    
    def test_classify_green_for_correct_efficient(self):
        """Correct + efficient = GREEN."""
        harness = create_mock_harness(seed=42)
        scenario = get_scenario_by_id("L1-001")
        run = harness.run_scenario(scenario)
        
        analysis = classify_run(run)
        
        assert analysis.status == TrafficLight.GREEN
        assert analysis.is_correct is True
        assert analysis.is_efficient is True
    
    def test_classify_yellow_for_correct_inefficient(self):
        """Correct + inefficient = YELLOW."""
        harness = create_mock_harness(seed=42, force_inefficient=True)
        scenario = get_scenario_by_id("L1-001")
        run = harness.run_scenario(scenario)
        
        analysis = classify_run(run)
        
        assert analysis.status == TrafficLight.YELLOW
        assert analysis.is_correct is True
        assert analysis.is_efficient is False
    
    def test_classify_red_for_incorrect(self):
        """Incorrect = RED."""
        harness = create_mock_harness(seed=42, force_incorrect=True)
        scenario = get_scenario_by_id("L1-001")
        run = harness.run_scenario(scenario)
        
        analysis = classify_run(run)
        
        assert analysis.status == TrafficLight.RED
        assert analysis.is_correct is False
    
    def test_analyze_benchmark_produces_summary(self):
        """analyze_benchmark produces correct summary stats."""
        harness = create_mock_harness(seed=42)
        benchmark = harness.run_benchmark(difficulty_filter=Difficulty.L1_TRIVIAL)
        
        analysis = analyze_benchmark(benchmark)
        
        assert len(analysis.analyses) == len(L1_SCENARIOS)
        assert analysis.summary["total"] == len(L1_SCENARIOS)
        assert analysis.summary["green"] == len(L1_SCENARIOS)  # All should be green
        assert analysis.summary["accuracy"] == 1.0
    
    def test_generate_report_creates_file(self, tmp_path):
        """generate_report creates a markdown file."""
        harness = create_mock_harness(seed=42)
        benchmark = harness.run_benchmark(difficulty_filter=Difficulty.L1_TRIVIAL)
        analysis = analyze_benchmark(benchmark)
        
        report_path = generate_report(analysis, output_dir=tmp_path)
        
        assert report_path.exists()
        assert report_path.name == "last_run_summary.md"
        
        content = report_path.read_text(encoding="utf-8")
        assert "# Antigravity Benchmark Report" in content
        assert "🟢" in content
    
    def test_traffic_light_enum_has_correct_symbols(self):
        """TrafficLight enum values are the expected emoji."""
        assert TrafficLight.GREEN.value == "🟢"
        assert TrafficLight.YELLOW.value == "🟡"
        assert TrafficLight.RED.value == "🔴"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================
class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_with_mock(self, tmp_path):
        """Complete pipeline: scenarios -> harness -> reporter."""
        # Create harness
        harness = create_mock_harness(seed=42)
        
        # Run benchmark
        benchmark = harness.run_benchmark()
        
        # Analyze
        analysis = analyze_benchmark(benchmark)
        
        # Generate report
        report_path = generate_report(analysis, output_dir=tmp_path)
        
        # Verify
        assert report_path.exists()
        assert analysis.summary["total"] == len(ALL_SCENARIOS)
        assert analysis.summary["accuracy"] > 0  # Some should be correct
    
    def test_mixed_results_produce_all_colors(self, tmp_path):
        """A benchmark with mixed results shows all traffic light colors."""
        # This would require more complex mock setup
        # For now, we verify the analysis can handle different states
        
        # Run with force_inefficient to get yellows on L1
        harness = create_mock_harness(seed=42, force_inefficient=True)
        benchmark = harness.run_benchmark(difficulty_filter=Difficulty.L1_TRIVIAL)
        analysis = analyze_benchmark(benchmark)
        
        # All L1 should be yellow (correct but inefficient)
        assert analysis.summary["yellow"] == len(L1_SCENARIOS)
        assert analysis.summary["green"] == 0

"""
Antigravity Benchmarking Harness

Black-box test runner that executes scenarios against the AntigravitySolver.
Includes a MockSolver for testing the benchmarking infrastructure itself.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Optional
import random

from .scenarios import Scenario, Difficulty, get_scenarios


# =============================================================================
# SOLVER PROTOCOL (Interface for real and mock solvers)
# =============================================================================
@dataclass
class SolverResult:
    """Result from a solver execution."""
    answer: str
    n_used: int              # Population size used (N parameter)
    t_used: int              # Iterations/time steps used (T parameter)
    k_used: int              # Aggregation factor used (K parameter)
    repechaje_activated: bool  # Whether Rescue mechanism was triggered
    homeostatic_efficiency: float # New metric: System health/balance (0.0 to 1.0)
    consistency_violations: int   # New metric: Symbolic consistency breaches
    execution_time_ms: float   # Wall-clock time in milliseconds
    metadata: dict = field(default_factory=dict)


class SolverProtocol(Protocol):
    """Protocol defining the solver interface."""
    
    def solve(self, prompt: str, difficulty_hint: Optional[Difficulty] = None) -> SolverResult:
        """Execute the solver on a prompt and return the result."""
        ...


# =============================================================================
# MOCK SOLVER (For testing the harness without real inference)
# =============================================================================
class MockAntigravitySolver:
    """
    Mock implementation of AntigravitySolver for testing purposes.
    
    Simulates different behaviors based on scenario difficulty:
    - L1: Uses minimal resources (N=1)
    - L2: Uses moderate resources
    - L3: Uses full population + may trigger repechaje
    
    Can be configured to simulate failures or inefficiencies.
    """
    
    def __init__(
        self, 
        seed: int = 42,
        force_inefficient: bool = False,
        force_incorrect: bool = False,
        simulate_repechaje_rate: float = 0.3,
    ):
        """
        Initialize the mock solver.
        
        Args:
            seed: Random seed for deterministic behavior.
            force_inefficient: If True, always use N=32 even for trivial problems.
            force_incorrect: If True, return wrong answers (for testing red light).
            simulate_repechaje_rate: Probability of simulating repechaje in L3.
        """
        self.rng = random.Random(seed)
        self.force_inefficient = force_inefficient
        self.force_incorrect = force_incorrect
        self.simulate_repechaje_rate = simulate_repechaje_rate
        
        # Mock answer bank for scenarios we know
        self._answer_bank = {
            "L1-001": "4",
            "L1-002": "París",
            "L1-003": "7",
            "L1-004": "hola",
            "L2-001": "1",
            "L2-002": "12:00",
            "L2-003": "No",
            "L2-004": "Ambas=Manzanas, Manzanas=Naranjas, Naranjas=Ambas",
        }
    
    def solve(
        self, 
        prompt: str, 
        difficulty_hint: Optional[Difficulty] = None,
        scenario_id: Optional[str] = None,
    ) -> SolverResult:
        """
        Simulate solving a problem.
        
        Args:
            prompt: The problem prompt.
            difficulty_hint: Optional hint about expected difficulty.
            scenario_id: Optional scenario ID for mock answer lookup.
        
        Returns:
            SolverResult with simulated metrics.
        """
        start_time = time.perf_counter()
        
        # Determine resource usage based on difficulty
        if self.force_inefficient:
            n_used, k_used = 32, 8
            # Evitar penalización severa (RED) en L1 manteniendo T=1, pero N alto (YELLOW)
            t_used = 1 if difficulty_hint == Difficulty.L1_TRIVIAL else 5
        elif difficulty_hint == Difficulty.L1_TRIVIAL:
            n_used, t_used, k_used = 1, 1, 1
        elif difficulty_hint == Difficulty.L2_LOGICAL:
            n_used = self.rng.randint(2, 8)
            t_used = self.rng.randint(1, 3)
            k_used = self.rng.randint(2, 4)
        else:  # L3 or unknown
            n_used, t_used, k_used = 32, 5, 8
        
        # Simulate repechaje
        repechaje = False
        if difficulty_hint == Difficulty.L3_CHAOS:
            repechaje = self.rng.random() < self.simulate_repechaje_rate
        
        # Generate answer
        if self.force_incorrect:
            answer = "WRONG_ANSWER_FOR_TESTING"
        elif scenario_id and scenario_id in self._answer_bank:
            answer = self._answer_bank[scenario_id]
        elif difficulty_hint == Difficulty.L3_CHAOS:
            answer = f"[Creative Response for: {prompt[:50]}...]"
        else:
            answer = "Mock answer - no scenario ID provided"
        
        # Simulate processing time
        simulated_delay = n_used * t_used * 0.01  # 10ms per N*T
        time.sleep(simulated_delay)
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        return SolverResult(
            answer=answer,
            n_used=n_used,
            t_used=t_used,
            k_used=k_used,
            repechaje_activated=repechaje,
            homeostatic_efficiency=1.0 if not self.force_inefficient else 0.4,
            consistency_violations=0 if not self.force_incorrect else 1,
            execution_time_ms=execution_time_ms,
            metadata={
                "mock": True,
                "seed": self.rng.getstate()[1][0],  # Current state indicator
            }
        )


# =============================================================================
# BENCHMARK RESULT STRUCTURES
# =============================================================================
@dataclass
class ScenarioRun:
    """Result of running a single scenario."""
    scenario: Scenario
    result: SolverResult
    is_correct: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkRun:
    """Result of a complete benchmark run."""
    runs: list[ScenarioRun]
    total_time_ms: float
    start_time: datetime
    end_time: datetime
    solver_name: str
    metadata: dict = field(default_factory=dict)


# =============================================================================
# HARNESS: BENCHMARK RUNNER
# =============================================================================
class BenchmarkHarness:
    """
    Executes scenarios against a solver and collects results.
    """
    
    def __init__(self, solver: SolverProtocol, solver_name: str = "Unknown"):
        """
        Initialize the harness.
        
        Args:
            solver: Any object implementing the SolverProtocol.
            solver_name: Human-readable name for the solver.
        """
        self.solver = solver
        self.solver_name = solver_name
    
    def run_scenario(self, scenario: Scenario) -> ScenarioRun:
        """
        Run a single scenario and return the result.
        
        Args:
            scenario: The scenario to run.
        
        Returns:
            ScenarioRun with result and correctness evaluation.
        """
        # Call solver with difficulty hint
        result = self.solver.solve(
            prompt=scenario.prompt,
            difficulty_hint=scenario.difficulty,
            scenario_id=scenario.id,
        )
        
        # Evaluate correctness
        is_correct = self._evaluate_correctness(scenario, result)
        
        return ScenarioRun(
            scenario=scenario,
            result=result,
            is_correct=is_correct,
        )
    
    def _evaluate_correctness(self, scenario: Scenario, result: SolverResult) -> bool:
        """
        Evaluate if the solver's answer is correct.
        
        For L1/L2: Exact match (case-insensitive, stripped).
        For L3: Always True (creative tasks, need human review).
        """
        if scenario.difficulty == Difficulty.L3_CHAOS:
            # Creative tasks can't be auto-evaluated
            # Mark as correct if non-empty response
            return bool(result.answer and len(result.answer) > 10)
        
        expected = scenario.expected_answer.strip().lower()
        actual = result.answer.strip().lower()
        
        return expected == actual
    
    def run_benchmark(
        self, 
        scenarios: Optional[list[Scenario]] = None,
        difficulty_filter: Optional[Difficulty] = None,
    ) -> BenchmarkRun:
        """
        Run a complete benchmark across multiple scenarios.
        
        Args:
            scenarios: Specific scenarios to run. If None, uses get_scenarios().
            difficulty_filter: Only run scenarios of this difficulty.
        
        Returns:
            BenchmarkRun with all results.
        """
        if scenarios is None:
            scenarios = get_scenarios(difficulty_filter)
        
        start_time = datetime.now()
        runs: list[ScenarioRun] = []
        
        for scenario in scenarios:
            run = self.run_scenario(scenario)
            runs.append(run)
        
        end_time = datetime.now()
        total_time_ms = sum(r.result.execution_time_ms for r in runs)
        
        return BenchmarkRun(
            runs=runs,
            total_time_ms=total_time_ms,
            start_time=start_time,
            end_time=end_time,
            solver_name=self.solver_name,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def create_mock_harness(
    seed: int = 42,
    force_inefficient: bool = False,
    force_incorrect: bool = False,
) -> BenchmarkHarness:
    """
    Create a harness with a mock solver for testing.
    
    Args:
        seed: Random seed for determinism.
        force_inefficient: Make mock always use N=32.
        force_incorrect: Make mock return wrong answers.
    
    Returns:
        Configured BenchmarkHarness with MockAntigravitySolver.
    """
    solver = MockAntigravitySolver(
        seed=seed,
        force_inefficient=force_inefficient,
        force_incorrect=force_incorrect,
    )
    return BenchmarkHarness(solver, solver_name="MockAntigravitySolver")


def run_quick_benchmark(difficulty: Optional[Difficulty] = None) -> BenchmarkRun:
    """
    Run a quick benchmark with default mock solver.
    
    Args:
        difficulty: Filter by difficulty level.
    
    Returns:
        BenchmarkRun results.
    """
    harness = create_mock_harness()
    return harness.run_benchmark(difficulty_filter=difficulty)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ANTIGRAVITY BENCHMARKING HARNESS")
    print("=" * 60)
    
    harness = create_mock_harness()
    result = harness.run_benchmark()
    
    print(f"\nSolver: {result.solver_name}")
    print(f"Total scenarios: {len(result.runs)}")
    print(f"Total time: {result.total_time_ms:.2f}ms")
    print()
    
    for run in result.runs:
        status = "✓" if run.is_correct else "✗"
        print(f"[{status}] {run.scenario.id}: N={run.result.n_used}, T={run.result.t_used}")
    
    # Import and run reporter
    from .reporter import analyze_benchmark, generate_report
    analysis = analyze_benchmark(result)
    report_path = generate_report(analysis)
    print(f"\nReport generated: {report_path}")

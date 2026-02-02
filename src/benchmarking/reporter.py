"""
Antigravity Benchmarking Reporter

Analyzes benchmark results and generates a report with Traffic Light grading:
- 🟢 GREEN: Correct answer + efficient resource usage
- 🟡 YELLOW: Correct answer + inefficient (overkill resources)
- 🔴 RED: Incorrect answer OR canon/symbolic violation
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import os

from .harness import BenchmarkRun, ScenarioRun
from .scenarios import Difficulty


class TrafficLight(Enum):
    """Traffic light status for scenario evaluation."""
    GREEN = "🟢"    # Correct + Efficient
    YELLOW = "🟡"   # Correct + Inefficient  
    RED = "🔴"      # Incorrect or Canon Violation


@dataclass
class ScenarioAnalysis:
    """Analysis of a single scenario run."""
    scenario_id: str
    status: TrafficLight
    is_correct: bool
    is_efficient: bool
    n_used: int
    t_used: int
    max_efficient_n: int
    max_efficient_t: int
    repechaje_used: bool
    homeostatic_efficiency: float
    consistency_violations: int
    improvement_per_sec: float
    execution_time_ms: float
    notes: str = ""


@dataclass  
class BenchmarkAnalysis:
    """Complete analysis of a benchmark run."""
    analyses: list[ScenarioAnalysis]
    summary: dict
    timestamp: datetime = field(default_factory=datetime.now)
    solver_name: str = ""


def classify_run(run: ScenarioRun) -> ScenarioAnalysis:
    """
    Classify a scenario run with traffic light logic.
    
    Logic:
    - RED: Incorrect answer
    - GREEN: Correct + N <= max_efficient_n AND T <= max_efficient_t
    - YELLOW: Correct but exceeded efficiency thresholds
    
    Args:
        run: The scenario run to analyze.
    
    Returns:
        ScenarioAnalysis with classification.
    """
    scenario = run.scenario
    result = run.result
    
    # Check efficiency
    is_n_efficient = result.n_used <= scenario.max_efficient_n
    is_t_efficient = result.t_used <= scenario.max_efficient_t
    is_efficient = is_n_efficient and is_t_efficient
    
    # Severe penalty for trivial tasks
    if scenario.difficulty == Difficulty.L1_TRIVIAL and result.t_used > 1:
        status = TrafficLight.RED
        notes = "SEVERE: T>1 on Trivial Task"
    # Consistency violations are automatic failures
    elif result.consistency_violations > 0:
        status = TrafficLight.RED
        notes = f"Consistency Violation ({result.consistency_violations})"
    # Standard traffic light logic
    elif not run.is_correct:
        status = TrafficLight.RED
        notes = "Incorrect answer"
    elif not is_efficient:
        status = TrafficLight.YELLOW
        inefficiencies = []
        if not is_n_efficient:
            inefficiencies.append(
                f"N={result.n_used} > max={scenario.max_efficient_n}"
            )
        if not is_t_efficient:
            inefficiencies.append(
                f"T={result.t_used} > max={scenario.max_efficient_t}"
            )
        notes = "Overkill: " + ", ".join(inefficiencies)
    else:
        status = TrafficLight.GREEN
        notes = "Optimal"
    
    # Calculate Improvement per Second
    # Formula: (RSA_Quality - Base_Quality) / (Time_RSA - Base_Time)
    # Assumptions:
    # - Base model (non-RSA) gets L1 correct (Q=1) instantly (T=0.1s)
    # - Base model fails L2/L3 (Q=0) quickly (T=0.5s)
    rsa_quality = 1.0 if run.is_correct else 0.0
    
    if scenario.difficulty == Difficulty.L1_TRIVIAL:
        base_quality = 1.0
        base_time_s = 0.1
    else:
        base_quality = 0.0
        base_time_s = 0.5
        
    rsa_time_s = result.execution_time_ms / 1000.0
    time_delta = rsa_time_s - base_time_s
    
    if time_delta == 0:
        imp_per_sec = 0.0
    else:
        imp_per_sec = (rsa_quality - base_quality) / time_delta

    return ScenarioAnalysis(
        scenario_id=scenario.id,
        status=status,
        is_correct=run.is_correct,
        is_efficient=is_efficient,
        n_used=result.n_used,
        t_used=result.t_used,
        max_efficient_n=scenario.max_efficient_n,
        max_efficient_t=scenario.max_efficient_t,
        repechaje_used=result.repechaje_activated,
        homeostatic_efficiency=result.homeostatic_efficiency,
        consistency_violations=result.consistency_violations,
        improvement_per_sec=imp_per_sec,
        execution_time_ms=result.execution_time_ms,
        notes=notes,
    )


def analyze_benchmark(benchmark: BenchmarkRun) -> BenchmarkAnalysis:
    """
    Analyze a complete benchmark run.
    
    Args:
        benchmark: The benchmark run to analyze.
    
    Returns:
        BenchmarkAnalysis with all classifications and summary.
    """
    analyses = [classify_run(run) for run in benchmark.runs]
    
    # Calculate summary statistics
    total = len(analyses)
    green_count = sum(1 for a in analyses if a.status == TrafficLight.GREEN)
    yellow_count = sum(1 for a in analyses if a.status == TrafficLight.YELLOW)
    red_count = sum(1 for a in analyses if a.status == TrafficLight.RED)
    
    correct_count = sum(1 for a in analyses if a.is_correct)
    efficient_count = sum(1 for a in analyses if a.is_efficient)
    repechaje_count = sum(1 for a in analyses if a.repechaje_used)
    
    # Group by difficulty
    by_difficulty = {
        Difficulty.L1_TRIVIAL: {"green": 0, "yellow": 0, "red": 0, "total": 0},
        Difficulty.L2_LOGICAL: {"green": 0, "yellow": 0, "red": 0, "total": 0},
        Difficulty.L3_CHAOS: {"green": 0, "yellow": 0, "red": 0, "total": 0},
    }
    
    for run, analysis in zip(benchmark.runs, analyses):
        diff = run.scenario.difficulty
        by_difficulty[diff]["total"] += 1
        if analysis.status == TrafficLight.GREEN:
            by_difficulty[diff]["green"] += 1
        elif analysis.status == TrafficLight.YELLOW:
            by_difficulty[diff]["yellow"] += 1
        else:
            by_difficulty[diff]["red"] += 1
    
    summary = {
        "total": total,
        "green": green_count,
        "yellow": yellow_count,
        "red": red_count,
        "correct": correct_count,
        "efficient": efficient_count,
        "repechaje_activations": repechaje_count,
        "accuracy": correct_count / total if total > 0 else 0,
        "consistency_violations": sum(a.consistency_violations for a in analyses),
        "avg_homeostatic_efficiency": sum(a.homeostatic_efficiency for a in analyses) / total if total > 0 else 0,
        "avg_improvement_per_sec": sum(a.improvement_per_sec for a in analyses) / total if total > 0 else 0,
        "efficiency_rate": efficient_count / total if total > 0 else 0,
        "total_time_ms": benchmark.total_time_ms,
        "by_difficulty": {
            "L1_TRIVIAL": by_difficulty[Difficulty.L1_TRIVIAL],
            "L2_LOGICAL": by_difficulty[Difficulty.L2_LOGICAL],
            "L3_CHAOS": by_difficulty[Difficulty.L3_CHAOS],
        },
    }
    
    return BenchmarkAnalysis(
        analyses=analyses,
        summary=summary,
        timestamp=benchmark.end_time,
        solver_name=benchmark.solver_name,
    )


def generate_report(
    analysis: BenchmarkAnalysis,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate a markdown report from the analysis.
    
    Args:
        analysis: The benchmark analysis.
        output_dir: Where to save the report. Defaults to ./results/
    
    Returns:
        Path to the generated report file.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "last_run_summary.md"
    
    # Build report content
    s = analysis.summary
    lines = [
        "# Antigravity Benchmark Report",
        "",
        f"**Solver:** {analysis.solver_name}",
        f"**Timestamp:** {analysis.timestamp.isoformat()}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Scenarios | {s['total']} |",
        f"| 🟢 Green (Optimal) | {s['green']} |",
        f"| 🟡 Yellow (Overkill) | {s['yellow']} |",
        f"| 🔴 Red (Incorrect) | {s['red']} |",
        f"| Accuracy | {s['accuracy']:.1%} |",
        f"| Efficiency Rate | {s['efficiency_rate']:.1%} |",
        f"| Repechaje Activations | {s['repechaje_activations']} |",
        f"| Consistency Violations | {s['consistency_violations']} |",
        f"| Avg Homeostatic Eff | {s['avg_homeostatic_efficiency']:.2f} |",
        f"| Avg Imp/Sec | {s['avg_improvement_per_sec']:.2f} |",
        f"| Total Time | {s['total_time_ms']:.2f}ms |",
        "",
        "---",
        "",
        "## By Difficulty",
        "",
        "| Level | 🟢 | 🟡 | 🔴 | Total |",
        "|-------|----|----|-----|-------|",
    ]
    
    for level, data in s["by_difficulty"].items():
        lines.append(
            f"| {level} | {data['green']} | {data['yellow']} | {data['red']} | {data['total']} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ])
    
    for a in analysis.analyses:
        lines.append(
            f"- {a.status.value} **{a.scenario_id}**: N={a.n_used}, T={a.t_used} "
            f"(max N={a.max_efficient_n}, T={a.max_efficient_t}) "
            f"Vio={a.consistency_violations} Eff={a.homeostatic_efficiency:.2f} "
            f"Imp/s={a.improvement_per_sec:.2f} "
            f"- {a.notes}"
        )
        if a.repechaje_used:
            lines.append(f"  - ⚡ Repechaje activated")
    
    lines.extend([
        "",
        "---",
        "",
        "## Interpretation Guide",
        "",
        "- **🟢 GREEN**: The L_kn (Manager) correctly allocated minimal resources.",
        "- **🟡 YELLOW**: Correct answer, but wasted compute (used N=32 for 2+2).",
        "- **🔴 RED**: Wrong answer, Consistency Violation, or Severe Inefficiency on Trivial Task.",
        "",
        "> This report is auto-generated by the Antigravity Benchmarking System.",
    ])
    
    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return report_path


def print_summary(analysis: BenchmarkAnalysis) -> None:
    """Print a quick summary to console."""
    s = analysis.summary
    print("\n" + "=" * 50)
    print("ANTIGRAVITY BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Solver: {analysis.solver_name}")
    print(f"Total: {s['total']} scenarios")
    print()
    print(f"  🟢 GREEN (Optimal):   {s['green']}")
    print(f"  🟡 YELLOW (Overkill): {s['yellow']}")
    print(f"  🔴 RED (Incorrect):   {s['red']}")
    print()
    print(f"Accuracy: {s['accuracy']:.1%}")
    print(f"Efficiency: {s['efficiency_rate']:.1%}")
    print(f"Repechaje Activations: {s['repechaje_activations']}")
    print("=" * 50)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Run a quick benchmark and generate report
    from .harness import create_mock_harness
    
    print("Running benchmark with mock solver...")
    harness = create_mock_harness()
    benchmark = harness.run_benchmark()
    
    print("Analyzing results...")
    analysis = analyze_benchmark(benchmark)
    
    print_summary(analysis)
    
    report_path = generate_report(analysis)
    print(f"\nReport saved to: {report_path}")

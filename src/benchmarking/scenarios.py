"""
Antigravity Benchmarking Scenarios

Defines test cases of varying difficulty to evaluate the AntigravitySolver:
- L1 (Trivial): Should require minimal resources (N=1, T=1)
- L2 (Logical): Requires reasoning, moderate resources
- L3 (Chaos/Creative): Requires full population (N=32) and possibly Repechaje
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Difficulty(Enum):
    """Difficulty levels for scenarios."""
    L1_TRIVIAL = 1   # Arithmetic, simple facts -> expect N=1
    L2_LOGICAL = 2   # Reasoning puzzles -> expect moderate N
    L3_CHAOS = 3     # Creative/Paradoxes -> expect N=32, Repechaje


@dataclass
class Scenario:
    """A single test scenario for the benchmarking harness."""
    id: str
    prompt: str
    difficulty: Difficulty
    expected_answer: str
    # Resource expectations (for traffic light logic)
    max_efficient_n: int = 1  # Max N to be considered "efficient" for this problem
    max_efficient_t: int = 1  # Max T (time steps/iterations) 
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None


# =============================================================================
# LEVEL 1: TRIVIAL SCENARIOS
# Expected behavior: N=1, T=1 (no population, single pass)
# =============================================================================
L1_SCENARIOS = [
    Scenario(
        id="L1-001",
        prompt="¿Cuánto es 2 + 2?",
        difficulty=Difficulty.L1_TRIVIAL,
        expected_answer="4",
        max_efficient_n=1,
        max_efficient_t=1,
        tags=["arithmetic", "trivial"],
    ),
    Scenario(
        id="L1-002",
        prompt="¿Cuál es la capital de Francia?",
        difficulty=Difficulty.L1_TRIVIAL,
        expected_answer="París",
        max_efficient_n=1,
        max_efficient_t=1,
        tags=["geography", "trivial"],
    ),
    Scenario(
        id="L1-003",
        prompt="¿Cuántos días tiene una semana?",
        difficulty=Difficulty.L1_TRIVIAL,
        expected_answer="7",
        max_efficient_n=1,
        max_efficient_t=1,
        tags=["facts", "trivial"],
    ),
    Scenario(
        id="L1-004",
        prompt="Traduce 'hello' al español.",
        difficulty=Difficulty.L1_TRIVIAL,
        expected_answer="hola",
        max_efficient_n=1,
        max_efficient_t=1,
        tags=["translation", "trivial"],
    ),
]


# =============================================================================
# LEVEL 2: LOGICAL SCENARIOS
# Expected behavior: Moderate N (2-8), possibly T>1 for reasoning chains
# =============================================================================
L2_SCENARIOS = [
    Scenario(
        id="L2-001",
        prompt=(
            "Sally tiene 3 hermanos. Cada hermano tiene 2 hermanas. "
            "¿Cuántas hermanas tiene Sally?"
        ),
        difficulty=Difficulty.L2_LOGICAL,
        expected_answer="1",  # Sally + 1 hermana = 2 hermanas para los hermanos
        max_efficient_n=4,
        max_efficient_t=2,
        tags=["logic", "family"],
        notes="Classic trick question. Sally IS one of the sisters.",
    ),
    Scenario(
        id="L2-002",
        prompt=(
            "Un tren sale de Madrid a las 9:00 a 100 km/h hacia Barcelona. "
            "Otro tren sale de Barcelona a las 10:00 a 150 km/h hacia Madrid. "
            "La distancia es 600 km. ¿A qué hora se cruzan?"
        ),
        difficulty=Difficulty.L2_LOGICAL,
        expected_answer="12:00",  # Primer tren recorre 100km en 1h, luego 250km/h combinados
        max_efficient_n=8,
        max_efficient_t=3,
        tags=["math", "physics", "trains"],
    ),
    Scenario(
        id="L2-003",
        prompt=(
            "Si llueve, el suelo está mojado. El suelo está mojado. "
            "¿Podemos concluir que llovió?"
        ),
        difficulty=Difficulty.L2_LOGICAL,
        expected_answer="No",  # Affirming the consequent fallacy
        max_efficient_n=4,
        max_efficient_t=2,
        tags=["logic", "fallacy"],
        notes="Affirming the consequent - other causes possible (sprinklers, etc.)",
    ),
    Scenario(
        id="L2-004",
        prompt=(
            "Tengo 3 cajas: una tiene manzanas, otra naranjas, otra ambas. "
            "Las etiquetas están TODAS MAL. Saco una fruta de la caja 'Ambas'. "
            "Sale una manzana. ¿Qué contiene cada caja?"
        ),
        difficulty=Difficulty.L2_LOGICAL,
        expected_answer="Ambas=Manzanas, Manzanas=Naranjas, Naranjas=Ambas",
        max_efficient_n=8,
        max_efficient_t=3,
        tags=["logic", "puzzle", "deduction"],
    ),
]


# =============================================================================
# LEVEL 3: CHAOS / CREATIVE SCENARIOS
# Expected behavior: Full population (N=32), multiple iterations, Repechaje likely
# =============================================================================
L3_SCENARIOS = [
    Scenario(
        id="L3-001",
        prompt=(
            "Diseña un sistema económico para una colonia en Marte que "
            "evite los problemas del capitalismo y el comunismo. "
            "Describe los mecanismos clave en 3 párrafos."
        ),
        difficulty=Difficulty.L3_CHAOS,
        expected_answer="",  # Creative - no single answer, evaluated by structure
        max_efficient_n=32,
        max_efficient_t=5,
        tags=["creative", "design", "economics"],
        notes="No single correct answer. Evaluate coherence and creativity.",
    ),
    Scenario(
        id="L3-002",
        prompt=(
            "Paradoja: 'Esta oración es falsa.' Analiza la paradoja desde "
            "tres perspectivas filosóficas diferentes y propón una resolución."
        ),
        difficulty=Difficulty.L3_CHAOS,
        expected_answer="",  # Philosophical - evaluated by depth
        max_efficient_n=32,
        max_efficient_t=5,
        tags=["philosophy", "paradox", "logic"],
        notes="Looking for Tarski, Russell, paraconsistent logic mentions.",
    ),
    Scenario(
        id="L3-003",
        prompt=(
            "Escribe el primer capítulo de una novela donde el protagonista "
            "descubre que sus sueños predicen el futuro, pero solo cuando "
            "intenta evitar lo que soñó, se cumple."
        ),
        difficulty=Difficulty.L3_CHAOS,
        expected_answer="",  # Creative writing - evaluated by narrative quality
        max_efficient_n=32,
        max_efficient_t=5,
        tags=["creative", "writing", "narrative"],
        notes="Evaluate narrative structure, irony handling, prose quality.",
    ),
    Scenario(
        id="L3-004",
        prompt=(
            "Un trolley se dirige hacia 5 personas. Puedes desviarlo, pero "
            "matará a 1 persona. TWIST: La persona sola es tu clon perfecto. "
            "¿Qué haces y por qué? Argumenta desde ética utilitarista, "
            "deontológica y de la virtud."
        ),
        difficulty=Difficulty.L3_CHAOS,
        expected_answer="",  # Ethical - evaluated by argument quality
        max_efficient_n=32,
        max_efficient_t=5,
        tags=["ethics", "philosophy", "trolley"],
        notes="Looking for identity considerations, multi-framework analysis.",
    ),
]


# =============================================================================
# AGGREGATED SCENARIO SETS
# =============================================================================
ALL_SCENARIOS = L1_SCENARIOS + L2_SCENARIOS + L3_SCENARIOS

SCENARIOS_BY_DIFFICULTY = {
    Difficulty.L1_TRIVIAL: L1_SCENARIOS,
    Difficulty.L2_LOGICAL: L2_SCENARIOS,
    Difficulty.L3_CHAOS: L3_SCENARIOS,
}


def get_scenarios(difficulty: Optional[Difficulty] = None) -> list[Scenario]:
    """
    Retrieve scenarios, optionally filtered by difficulty.
    
    Args:
        difficulty: If provided, return only scenarios of this difficulty.
                   If None, return all scenarios.
    
    Returns:
        List of Scenario objects.
    """
    if difficulty is None:
        return ALL_SCENARIOS
    return SCENARIOS_BY_DIFFICULTY.get(difficulty, [])


def get_scenario_by_id(scenario_id: str) -> Optional[Scenario]:
    """Retrieve a specific scenario by its ID."""
    for scenario in ALL_SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    return None

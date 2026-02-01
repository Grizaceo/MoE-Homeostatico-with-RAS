"""
RSA Engine - Recursive Self-Aggregation con Estratificación y Repechaje.

Módulo core del motor RSA con modificaciones propietarias Antigravity:
- Estratificación semántica en lugar de random.sample()
- Mecanismo de Repechaje para outliers de alta calidad
"""

from src.rsa_engine.population import Candidate, PopulationManager
from src.rsa_engine.selection import stratified_sample, RepechageBuffer
from src.rsa_engine.solver import RSASolver, RSAConfig, RSAResult

__all__ = [
    "Candidate",
    "PopulationManager",
    "stratified_sample",
    "RepechageBuffer",
    "RSASolver",
    "RSAConfig",
    "RSAResult",
]

"""
RAS Core - Controlador Homeostático para RSA.

Sistema de Activación Reticular (RAS) para economía de la atención.
Decide dinámicamente los hiperparámetros del RSA Engine.
"""

from src.ras_core.manager import (
    RASController,
    QueryContext,
    RSABudget,
    BudgetProfile,
)

__all__ = [
    "RASController",
    "QueryContext",
    "RSABudget",
    "BudgetProfile",
]

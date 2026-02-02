"""
RSI Logger - Sistema de logging estructurado basado en Real/Simbólico/Imaginario.

Este logger se integra desde Fase 1 para registrar todas las operaciones
con tags RSI, permitiendo análisis posterior de comportamiento del sistema.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- log_level: Nivel mínimo de logging
- output_format: "jsonl" o "console"
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MetricTag(Enum):
    """
    Clasificación de métricas homeostáticas.
    
    - COMPUTE_CONSTRAINTS: Límites de hardware (VRAM, tiempo, errores)
    - LOGICAL_CONSISTENCY: Reglas, topología, contratos verificados
    - POPULATION_VARIANCE: Diversidad, entropía, representaciones
    
    Nota histórica: Originalmente RSI (Real/Simbólico/Imaginario)
    del framework lacaniano-computacional.
    """
    COMPUTE_CONSTRAINTS = "CC"
    LOGICAL_CONSISTENCY = "LC"
    POPULATION_VARIANCE = "PV"


# Alias de compatibilidad hacia atrás
RSITag = MetricTag
RSITag.REAL = MetricTag.COMPUTE_CONSTRAINTS
RSITag.SIMBOLICO = MetricTag.LOGICAL_CONSISTENCY
RSITag.IMAGINARIO = MetricTag.POPULATION_VARIANCE


@dataclass
class RSIEvent:
    """Evento estructurado con tag RSI."""
    
    timestamp: str
    tag: str  # RSITag value
    component: str  # Módulo que genera el evento
    action: str  # Acción específica
    details: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class RSILogger:
    """
    Logger estructurado con clasificación RSI.
    
    Uso:
        logger = RSILogger("ricci_cleaning")
        logger.log_real("connectivity_check", {"components": 1})
        logger.log_simbolico("threshold_applied", {"value": -0.5})
        logger.log_imaginario("visualization", {"format": "png"})
    """
    
    def __init__(
        self,
        component: str,
        output_path: Path | str | None = None,  # PARAM: ruta de salida
        console_output: bool = True,  # PARAM: imprimir a consola
    ):
        self.component = component
        self.output_path = Path(output_path) if output_path else None
        self.console_output = console_output
        self._events: list[RSIEvent] = []
        
        # Crear directorio si no existe
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _create_event(
        self,
        tag: MetricTag,
        action: str,
        details: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> RSIEvent:
        return RSIEvent(
            timestamp=datetime.now().isoformat(),
            tag=tag.value,
            component=self.component,
            action=action,
            details=details or {},
            metrics=metrics or {},
        )
    
    def _emit(self, event: RSIEvent) -> None:
        """Emite el evento a los outputs configurados."""
        self._events.append(event)
        
        if self.console_output:
            # Formato compacto para consola
            print(
                f"[{event.tag}] {event.component}.{event.action} "
                f"| {event.details} | metrics={event.metrics}"
            )
        
        if self.output_path:
            # JSONL para análisis posterior
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
    
    def log_compute_constraints(
        self,
        action: str,
        details: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log de COMPUTE_CONSTRAINTS: límites hardware, errores, timeouts.
        (Antes REAL)
        """
        event = self._create_event(MetricTag.COMPUTE_CONSTRAINTS, action, details, metrics)
        self._emit(event)
    
    def log_logical_consistency(
        self,
        action: str,
        details: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log de LOGICAL_CONSISTENCY: reglas, contratos, verificación.
        (Antes SIMBÓLICO)
        """
        event = self._create_event(MetricTag.LOGICAL_CONSISTENCY, action, details, metrics)
        self._emit(event)
    
    def log_population_variance(
        self,
        action: str,
        details: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log de POPULATION_VARIANCE: diversidad, entropía, representaciones.
        (Antes IMAGINARIO)
        """
        event = self._create_event(MetricTag.POPULATION_VARIANCE, action, details, metrics)
        self._emit(event)

    # Alias de compatibilidad
    log_real = log_compute_constraints
    log_simbolico = log_logical_consistency
    log_imaginario = log_population_variance
    
    def get_events(self, tag: MetricTag | None = None) -> list[RSIEvent]:
        """Obtiene eventos, opcionalmente filtrados por tag."""
        if tag is None:
            return self._events.copy()
        return [e for e in self._events if e.tag == tag.value]
    
    def get_summary(self) -> dict[str, int]:
        """Resumen de conteo de eventos por tag."""
        return {
            "CC": len(self.get_events(MetricTag.COMPUTE_CONSTRAINTS)),
            "LC": len(self.get_events(MetricTag.LOGICAL_CONSISTENCY)),
            "PV": len(self.get_events(MetricTag.POPULATION_VARIANCE)),
            "total": len(self._events),
        }
    
    def clear(self) -> None:
        """Limpia eventos en memoria (no afecta archivo)."""
        self._events.clear()


# Logger global por defecto (singleton pattern)
_default_logger: RSILogger | None = None


def get_logger(component: str = "default") -> RSILogger:
    """Obtiene o crea el logger global."""
    global _default_logger
    if _default_logger is None:
        _default_logger = RSILogger(component, console_output=True)
    return _default_logger


def set_default_logger(logger: RSILogger) -> None:
    """Establece el logger global."""
    global _default_logger
    _default_logger = logger

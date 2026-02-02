"""
Tests para el sistema RSI Logger.
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.verification.rsi_logger import (
    RSILogger,
    RSILogger,
    MetricTag,
    RSITag, # Alias
    RSIEvent,
    RSIEvent,
    get_logger,
)


class TestRSIEvent:
    """Tests para RSIEvent dataclass."""
    
    def test_to_dict(self):
        event = RSIEvent(
            timestamp="2026-02-01T12:00:00",
            tag="CC",
            component="test",
            action="check",
            details={"key": "value"},
            metrics={"latency": 0.5},
        )
        d = event.to_dict()
        assert d["tag"] == "CC"
        assert d["component"] == "test"
        assert d["details"]["key"] == "value"
    
    def test_to_json(self):
        event = RSIEvent(
            timestamp="2026-02-01T12:00:00",
            tag="LC",
            component="verify",
            action="contract",
        )
        j = event.to_json()
        parsed = json.loads(j)
        assert parsed["tag"] == "LC"


class TestRSILogger:
    """Tests para RSILogger."""
    
    def test_log_real(self):
        logger = RSILogger("test_component", console_output=False)
        logger.log_real("error_detected", {"error": "NaN"})
        
        events = logger.get_events(RSITag.REAL)
        assert len(events) == 1
        assert events[0].action == "error_detected"
    
    def test_log_simbolico(self):
        logger = RSILogger("test_component", console_output=False)
        logger.log_simbolico("threshold_check", {"value": -0.5})
        
        events = logger.get_events(RSITag.SIMBOLICO)
        assert len(events) == 1
        assert events[0].details["value"] == -0.5
    
    def test_log_imaginario(self):
        logger = RSILogger("test_component", console_output=False)
        logger.log_imaginario("embedding_created", metrics={"dim": 32})
        
        events = logger.get_events(RSITag.IMAGINARIO)
        assert len(events) == 1
        assert events[0].metrics["dim"] == 32
    
    def test_get_summary(self):
        logger = RSILogger("test", console_output=False)
        logger.log_real("a")
        logger.log_real("b")
        logger.log_simbolico("c")
        logger.log_imaginario("d")
        logger.log_imaginario("e")
        logger.log_imaginario("f")
        
        summary = logger.get_summary()
        assert summary["CC"] == 2
        assert summary["LC"] == 1
        assert summary["PV"] == 3
        assert summary["total"] == 6
    
    def test_file_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_log.jsonl"
            logger = RSILogger("file_test", output_path=log_path, console_output=False)
            
            logger.log_real("event1", {"info": "test"})
            logger.log_simbolico("event2")
            
            # Verificar que el archivo existe y tiene contenido
            assert log_path.exists()
            
            with open(log_path, "r") as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            
            # Verificar que es JSON válido
            event1 = json.loads(lines[0])
            assert event1["tag"] == "CC"
            assert event1["action"] == "event1"
    
    def test_clear(self):
        logger = RSILogger("test", console_output=False)
        logger.log_real("a")
        logger.log_simbolico("b")
        
        assert logger.get_summary()["total"] == 2
        
        logger.clear()
        
        assert logger.get_summary()["total"] == 0
    
    def test_filter_by_tag(self):
        logger = RSILogger("test", console_output=False)
        logger.log_real("r1")
        logger.log_simbolico("s1")
        logger.log_real("r2")
        
        real_events = logger.get_events(RSITag.REAL)
        assert len(real_events) == 2
        
        simb_events = logger.get_events(RSITag.SIMBOLICO)
        assert len(simb_events) == 1
        
        all_events = logger.get_events()
        assert len(all_events) == 3

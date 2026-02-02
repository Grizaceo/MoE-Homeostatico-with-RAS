"""
Backend LLM para Single-Backbone RSA.

Este módulo implementa el generador unificado que usa UN SOLO modelo
cargado en VRAM, simulando "expertos" mediante cambio de system_role.

RESTRICCIÓN: RTX 4060 (8GB VRAM) - Todo en una sola instancia.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- model_id: Modelo a cargar
- max_new_tokens: Límite de tokens generados
- default_temperature: Temperatura base
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal
import numpy as np

from src.verification.rsi_logger import RSILogger


# ============== System Roles para Simular Expertos ==============

EXPERT_ROLES: dict[str, str] = {
    "default": (
        "You are a helpful AI assistant. Provide clear, accurate, "
        "and well-reasoned responses."
    ),
    "analyst": (
        "You are a critical analyst. Focus strictly on logic, evidence, "
        "and factual accuracy. Question assumptions and identify weaknesses."
    ),
    "creative": (
        "You are a creative thinker. Explore unconventional ideas, "
        "make unexpected connections, and propose novel solutions."
    ),
    "synthesizer": (
        "You are a master synthesizer. Your role is to combine multiple "
        "perspectives into a coherent, balanced, and comprehensive answer."
    ),
    "critic": (
        "You are a constructive critic. Find flaws, edge cases, and "
        "potential issues in arguments. Be thorough but fair."
    ),
    "code_expert": (
        "You are an expert software engineer. Focus on code quality, "
        "best practices, efficiency, and maintainability."
    ),
    "legal_expert": (
        "You are a legal expert. Analyze issues from a legal perspective, "
        "consider regulations, precedents, and compliance."
    ),
}


# ============== Interfaces ==============

class LLMBackend(Protocol):
    """Protocolo para backends LLM."""
    
    def generate(
        self,
        prompt: str,
        system_role: str = "default",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        """Genera una respuesta."""
        ...
    
    def aggregate(
        self,
        query: str,
        responses: list[str],
        temperature: float = 0.3,
    ) -> str:
        """Sintetiza múltiples respuestas."""
        ...


# ============== Mock Backend (para testing sin GPU) ==============

class MockSingleModelGenerator:
    """
    Backend mock para testing sin modelo real.
    
    Simula respuestas determinísticas basadas en inputs.
    """
    
    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._call_count = 0
        self._logger = RSILogger("mock_backend", console_output=False)
    
    def generate(
        self,
        prompt: str,
        system_role: str = "default",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        """Genera respuesta mock."""
        self._call_count += 1
        
        role_flavor = EXPERT_ROLES.get(system_role, EXPERT_ROLES["default"])[:30]
        response_hash = hash(prompt + system_role + str(self._call_count)) % 10000
        
        response = (
            f"[Mock #{response_hash}][Role: {system_role}] "
            f"Response with temp={temperature:.2f} for: {prompt[:50]}..."
        )
        
        self._logger.log_simbolico(
            "mock_generate",
            details={"role": system_role, "temp": temperature},
        )
        
        return response
    
    def generate_batch(
        self,
        prompts: list[str],
        system_role: str = "default",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Genera batch de respuestas."""
        return [
            self.generate(p, system_role, temperature, max_new_tokens)
            for p in prompts
        ]
    
    def aggregate(
        self,
        query: str,
        responses: list[str],
        temperature: float = 0.3,
    ) -> str:
        """Sintetiza respuestas mock."""
        self._call_count += 1
        
        return (
            f"[Mock Synthesis] Combined {len(responses)} responses "
            f"for query: {query[:40]}..."
        )


# ============== Real Backend (requiere transformers) ==============

class SingleModelGenerator:
    """
    Generador con UN SOLO modelo cargado en VRAM.
    
    Simula "expertos" cambiando el system_role dinámicamente.
    Optimizado para RTX 4060 (8GB VRAM).
    
    Backend soportados:
    - "transformers": HuggingFace transformers (default)
    - "vllm": vLLM para batching eficiente
    - "mock": Sin modelo real (testing)
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",  # PARAM
        backend_type: Literal["transformers", "vllm", "mock"] = "mock",
        device: str = "cuda",
        torch_dtype: str = "auto",  # PARAM: "float16", "bfloat16", "auto"
        max_new_tokens: int = 512,  # PARAM
        seed: int | None = None,
        logger: RSILogger | None = None,
    ):
        self.model_id = model_id
        self.backend_type = backend_type
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._seed = seed
        self._logger = logger or RSILogger("single_model", console_output=False)
        
        self._model = None
        self._tokenizer = None
        self._call_count = 0
        
        if backend_type == "mock":
            self._mock = MockSingleModelGenerator(seed=seed)
        else:
            self._mock = None
            self._load_model(torch_dtype)
    
    def _load_model(self, torch_dtype: str) -> None:
        """Carga el modelo en VRAM."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "auto": "auto",
            }
            
            self._logger.log_simbolico(
                "model_loading",
                details={"model_id": self.model_id, "device": self.device},
            )
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype_map.get(torch_dtype, "auto"),
                device_map=self.device,
                trust_remote_code=True,
            )
            
            # Establecer pad token si no existe
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._logger.log_simbolico(
                "model_loaded",
                details={
                    "model_id": self.model_id,
                    "dtype": str(self._model.dtype),
                },
            )
            
        except ImportError as e:
            raise ImportError(
                f"Backend '{self.backend_type}' requires transformers. "
                f"Install with: pip install transformers torch"
            ) from e
    
    def generate(
        self,
        prompt: str,
        system_role: str = "default",
        temperature: float = 0.7,
        max_new_tokens: int | None = None,
    ) -> str:
        """
        Genera una respuesta con el role especificado.
        
        Args:
            prompt: Prompt del usuario
            system_role: Key de EXPERT_ROLES o string custom
            temperature: Temperatura de sampling
            max_new_tokens: Override de max tokens
        
        Returns:
            Respuesta generada
        """
        if self._mock is not None:
            return self._mock.generate(prompt, system_role, temperature)
        
        self._call_count += 1
        max_tokens = max_new_tokens or self.max_new_tokens
        
        # Obtener system prompt
        system_prompt = EXPERT_ROLES.get(system_role, system_role)
        
        # Formatear mensajes
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        # Tokenizar con template de chat
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generar
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        
        # Decodificar solo la parte generada
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        self._logger.log_simbolico(
            "generation_completed",
            details={"role": system_role, "temp": temperature},
            metrics={"response_length": len(response)},
        )
        
        return response.strip()
    
    def generate_batch(
        self,
        prompts: list[str],
        system_role: str = "default",
        temperature: float = 0.7,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """
        Genera batch de respuestas.
        
        Nota: Con transformers vanilla, esto es secuencial.
        Para batching real, usar vLLM backend.
        """
        return [
            self.generate(p, system_role, temperature, max_new_tokens)
            for p in prompts
        ]
    
    def aggregate(
        self,
        query: str,
        responses: list[str],
        temperature: float = 0.3,
    ) -> str:
        """
        Sintetiza múltiples respuestas en una.
        
        Usa el role "synthesizer" para combinar perspectivas.
        """
        # Formatear respuestas numeradas
        formatted_responses = "\n\n".join(
            f"### Response {i+1}:\n{r}"
            for i, r in enumerate(responses)
        )
        
        aggregation_prompt = f"""You are synthesizing multiple AI responses into one coherent answer.

## Original Query:
{query}

## Responses to Synthesize:
{formatted_responses}

## Your Task:
Combine the best insights from all responses into a single, comprehensive answer. 
Resolve any contradictions and present a unified perspective.

## Synthesized Answer:"""
        
        return self.generate(
            prompt=aggregation_prompt,
            system_role="synthesizer",
            temperature=temperature,
        )
    
    def get_stats(self) -> dict:
        """Retorna estadísticas del backend."""
        return {
            "model_id": self.model_id,
            "backend_type": self.backend_type,
            "total_calls": self._call_count,
            "is_loaded": self._model is not None or self._mock is not None,
        }


# ============== Factory ==============

def create_backend(
    backend_type: Literal["transformers", "vllm", "mock"] = "mock",
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    **kwargs,
) -> SingleModelGenerator:
    """
    Factory para crear backend LLM.
    
    Args:
        backend_type: Tipo de backend
        model_id: ID del modelo (HuggingFace o local)
        **kwargs: Args adicionales para SingleModelGenerator
    
    Returns:
        Instancia configurada
    """
    return SingleModelGenerator(
        model_id=model_id,
        backend_type=backend_type,
        **kwargs,
    )

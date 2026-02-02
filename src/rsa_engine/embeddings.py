"""
Embeddings semánticos para RSA Engine.

Proveedor de embeddings usando sentence-transformers.
Ejecuta en CPU para no competir con LLM en VRAM.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- model_name: Modelo de embeddings a usar
- device: "cpu" o "cuda" (default: CPU para preservar VRAM)
"""

from __future__ import annotations

from typing import Protocol
import numpy as np

from src.verification.rsi_logger import RSILogger


# ============== Interface ==============

class EmbeddingProvider(Protocol):
    """Protocolo para proveedores de embeddings."""
    
    def embed(self, text: str) -> np.ndarray:
        """Genera embedding para un texto."""
        ...
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Genera embeddings para múltiples textos."""
        ...
    
    @property
    def dim(self) -> int:
        """Dimensión del embedding."""
        ...


# ============== Mock Provider ==============

class MockEmbeddingProvider:
    """
    Proveedor mock para testing sin modelo real.
    
    Genera embeddings determinísticos basados en hash.
    """
    
    def __init__(self, dim: int = 384, seed: int | None = None):
        self._dim = dim
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}
    
    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            # Embedding determinístico basado en hash
            text_hash = hash(text) % (2**32)
            local_rng = np.random.default_rng(text_hash)
            embedding = local_rng.standard_normal(self._dim)
            embedding = embedding / np.linalg.norm(embedding)
            self._cache[text] = embedding.astype(np.float32)
        return self._cache[text]
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])
    
    @property
    def dim(self) -> int:
        return self._dim


# ============== Sentence Transformers Provider ==============

class SentenceTransformerProvider:
    """
    Proveedor de embeddings usando sentence-transformers.
    
    Ejecuta en CPU por defecto para preservar VRAM para el LLM.
    
    Modelos recomendados (balance velocidad/calidad):
    - "all-MiniLM-L6-v2": 384 dims, muy rápido
    - "all-mpnet-base-v2": 768 dims, mejor calidad
    - "paraphrase-multilingual-MiniLM-L12-v2": Multilingüe
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # PARAM
        device: str = "cpu",  # PARAM: CPU para preservar VRAM
        normalize: bool = True,  # PARAM: Normalizar embeddings
        logger: RSILogger | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._logger = logger or RSILogger("embeddings", console_output=False)
        
        self._model = None
        self._dim: int | None = None
        self._call_count = 0
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Carga el modelo de embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self._logger.log_simbolico(
                "embedding_model_loading",
                details={"model": self.model_name, "device": self.device},
            )
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            
            # Obtener dimensión
            self._dim = self._model.get_sentence_embedding_dimension()
            
            self._logger.log_simbolico(
                "embedding_model_loaded",
                details={"dim": self._dim},
            )
            
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerProvider requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
    
    def embed(self, text: str) -> np.ndarray:
        """
        Genera embedding para un texto.
        
        Args:
            text: Texto a embeber
        
        Returns:
            Vector numpy normalizado
        """
        self._call_count += 1
        
        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Genera embeddings para múltiples textos (batch).
        
        Args:
            texts: Lista de textos
        
        Returns:
            Array numpy de shape (n_texts, dim)
        """
        self._call_count += len(texts)
        
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        
        return embeddings.astype(np.float32)
    
    @property
    def dim(self) -> int:
        """Dimensión del embedding."""
        if self._dim is None:
            raise RuntimeError("Model not loaded")
        return self._dim
    
    def get_stats(self) -> dict:
        """Retorna estadísticas del proveedor."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dim": self._dim,
            "total_calls": self._call_count,
        }


# ============== Factory ==============

def create_embedding_provider(
    provider_type: str = "mock",
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs,
) -> EmbeddingProvider:
    """
    Factory para crear proveedor de embeddings.
    
    Args:
        provider_type: "mock" o "sentence-transformers"
        model_name: Modelo para sentence-transformers
        **kwargs: Args adicionales
    
    Returns:
        Instancia de proveedor
    """
    if provider_type == "mock":
        return MockEmbeddingProvider(**kwargs)
    elif provider_type == "sentence-transformers":
        return SentenceTransformerProvider(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

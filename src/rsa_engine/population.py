"""
Gestión de población de candidatos para RSA.

Este módulo maneja la población de respuestas candidatas con sus embeddings
semánticos y métricas de curvatura. Utiliza NetworkX para el grafo de similitud.

VARIABLES AJUSTABLES (marcadas con # PARAM):
- similarity_threshold: Umbral para crear aristas en grafo de similitud
- default_embedding_dim: Dimensión por defecto de embeddings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np
import networkx as nx

from src.verification.rsi_logger import RSILogger


# ============== Interfaces ==============

class EmbeddingProvider(Protocol):
    """Protocolo para proveedores de embeddings semánticos."""
    
    def embed(self, text: str) -> np.ndarray:
        """Genera embedding para un texto."""
        ...
    
    @property
    def dim(self) -> int:
        """Dimensión del embedding."""
        ...


class MockEmbeddingProvider:
    """
    Proveedor de embeddings mock para testing.
    
    Genera embeddings determinísticos basados en hash del texto.
    """
    
    def __init__(self, dim: int = 64, seed: int | None = None):
        self._dim = dim
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}
    
    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            # Embedding determinístico basado en hash
            text_hash = hash(text) % (2**32)
            local_rng = np.random.default_rng(text_hash)
            embedding = local_rng.standard_normal(self._dim)
            embedding = embedding / np.linalg.norm(embedding)  # Normalizar
            self._cache[text] = embedding
        return self._cache[text]
    
    @property
    def dim(self) -> int:
        return self._dim


# ============== Dataclasses ==============

@dataclass
class Candidate:
    """
    Candidato en la población RSA.
    
    Representa una respuesta generada con su metadata semántica y topológica.
    
    Attributes:
        id: Identificador único del candidato
        response: Texto de la respuesta
        embedding: Vector semántico (normalizado)
        ricci_curvature: Curvatura local promedio (estabilidad topológica)
        score: Score de calidad/confianza asignado
        round_created: Ronda de RSA en que fue creado
        parent_ids: IDs de candidatos usados para agregar este
        metadata: Datos adicionales arbitrarios
    """
    id: int
    response: str
    embedding: np.ndarray
    ricci_curvature: float = 0.0
    score: float = 0.0
    round_created: int = 0
    parent_ids: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def semantic_distance(self, other: "Candidate") -> float:
        """
        Calcula distancia semántica (1 - similitud coseno).
        
        Returns:
            Distancia en [0, 2], donde 0 = idénticos, 2 = opuestos
        """
        similarity = np.dot(self.embedding, other.embedding)
        return 1.0 - similarity
    
    def semantic_similarity(self, other: "Candidate") -> float:
        """
        Calcula similitud coseno con otro candidato.
        
        Returns:
            Similitud en [-1, 1], donde 1 = idénticos
        """
        return float(np.dot(self.embedding, other.embedding))


# ============== Population Manager ==============

class PopulationManager:
    """
    Gestiona la población de candidatos con grafo NetworkX subyacente.
    
    El grafo de similitud conecta candidatos semánticamente cercanos,
    permitiendo cálculo eficiente de curvatura de Ricci local.
    
    Attributes:
        candidates: Diccionario id -> Candidate
        graph: Grafo de similitud (nodos = IDs, aristas = similitud > threshold)
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.5,  # PARAM: umbral para crear arista
        seed: int | None = None,
        logger: RSILogger | None = None,
    ):
        self.candidates: dict[int, Candidate] = {}
        self.graph: nx.Graph = nx.Graph()
        self.similarity_threshold = similarity_threshold
        self._next_id = 0
        self._rng = np.random.default_rng(seed)
        self._embedding_provider = embedding_provider or MockEmbeddingProvider(seed=seed)
        self._logger = logger or RSILogger("population", console_output=False)
    
    @property
    def size(self) -> int:
        """Número de candidatos en la población."""
        return len(self.candidates)
    
    def add_candidate(
        self,
        response: str,
        embedding: np.ndarray | None = None,
        round_created: int = 0,
        parent_ids: list[int] | None = None,
        score: float = 0.0,
        metadata: dict | None = None,
    ) -> int:
        """
        Añade un candidato a la población.
        
        Args:
            response: Texto de la respuesta
            embedding: Vector semántico (se computa si None)
            round_created: Ronda en que se creó
            parent_ids: IDs de padres (para candidatos agregados)
            score: Score inicial
            metadata: Datos adicionales
        
        Returns:
            ID del candidato creado
        """
        if embedding is None:
            embedding = self._embedding_provider.embed(response)
        
        # Normalizar embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        candidate = Candidate(
            id=self._next_id,
            response=response,
            embedding=embedding,
            round_created=round_created,
            parent_ids=parent_ids or [],
            score=score,
            metadata=metadata or {},
        )
        
        self.candidates[self._next_id] = candidate
        self.graph.add_node(self._next_id)
        
        # Conectar con candidatos similares
        self._update_graph_connections(self._next_id)
        
        self._next_id += 1
        return candidate.id
    
    def _update_graph_connections(self, new_id: int) -> None:
        """Actualiza conexiones del grafo para un nuevo candidato."""
        new_candidate = self.candidates[new_id]
        
        for other_id, other_candidate in self.candidates.items():
            if other_id == new_id:
                continue
            
            similarity = new_candidate.semantic_similarity(other_candidate)
            
            if similarity >= self.similarity_threshold:
                self.graph.add_edge(new_id, other_id, weight=similarity)
    
    def rebuild_similarity_graph(self, threshold: float | None = None) -> None:
        """
        Reconstruye el grafo de similitud desde cero.
        
        Args:
            threshold: Nuevo umbral (usa el actual si None)
        """
        if threshold is not None:
            self.similarity_threshold = threshold
        
        # Limpiar aristas
        self.graph.clear_edges()
        
        # Reconstruir
        ids = list(self.candidates.keys())
        for i, id_i in enumerate(ids):
            for id_j in ids[i + 1:]:
                similarity = self.candidates[id_i].semantic_similarity(
                    self.candidates[id_j]
                )
                if similarity >= self.similarity_threshold:
                    self.graph.add_edge(id_i, id_j, weight=similarity)
        
        self._logger.log_simbolico(
            "similarity_graph_rebuilt",
            details={
                "n_nodes": self.graph.number_of_nodes(),
                "n_edges": self.graph.number_of_edges(),
                "threshold": self.similarity_threshold,
            },
        )
    
    def compute_local_curvature(self, candidate_id: int) -> float:
        """
        Calcula curvatura de Ricci local para un candidato.
        
        Usa Forman-Ricci simplificado: F(e) = 4 - deg(u) - deg(v)
        Promedia sobre todas las aristas incidentes al nodo.
        
        Returns:
            Curvatura promedio (negativa = hiperbólico, positiva = esférico)
        """
        if candidate_id not in self.graph:
            return 0.0
        
        neighbors = list(self.graph.neighbors(candidate_id))
        if not neighbors:
            return 0.0
        
        deg_self = self.graph.degree(candidate_id)
        curvatures = []
        
        for neighbor in neighbors:
            deg_neighbor = self.graph.degree(neighbor)
            # Forman-Ricci: F(e) = 4 - deg(u) - deg(v)
            edge_curvature = 4 - deg_self - deg_neighbor
            curvatures.append(edge_curvature)
        
        avg_curvature = float(np.mean(curvatures))
        
        # Actualizar candidato
        self.candidates[candidate_id].ricci_curvature = avg_curvature
        
        return avg_curvature
    
    def update_all_curvatures(self) -> dict[int, float]:
        """
        Actualiza curvatura de todos los candidatos.
        
        Returns:
            Diccionario id -> curvatura
        """
        curvatures = {}
        for cid in self.candidates:
            curvatures[cid] = self.compute_local_curvature(cid)
        
        self._logger.log_simbolico(
            "curvatures_updated",
            metrics={
                "n_candidates": len(curvatures),
                "mean_curvature": float(np.mean(list(curvatures.values()))),
                "min_curvature": float(np.min(list(curvatures.values()))),
                "max_curvature": float(np.max(list(curvatures.values()))),
            },
        )
        
        return curvatures
    
    def get_expert_profile_embedding(self, expert_id: int) -> np.ndarray:
        """
        Obtiene el embedding del "perfil" de un experto.
        
        El perfil es simplemente el embedding del candidato experto.
        Podría extenderse a promedios de vecindario, etc.
        
        Args:
            expert_id: ID del candidato experto
        
        Returns:
            Embedding del experto
        """
        if expert_id not in self.candidates:
            raise KeyError(f"Experto {expert_id} no encontrado en población")
        
        return self.candidates[expert_id].embedding.copy()
    
    def get_candidates_by_round(self, round_num: int) -> list[Candidate]:
        """Obtiene candidatos creados en una ronda específica."""
        return [c for c in self.candidates.values() if c.round_created == round_num]
    
    def get_top_candidates(self, n: int, by: str = "score") -> list[Candidate]:
        """
        Obtiene los top N candidatos ordenados por métrica.
        
        Args:
            n: Número de candidatos
            by: Campo por el cual ordenar ("score", "ricci_curvature")
        
        Returns:
            Lista de candidatos ordenados descendentemente
        """
        sorted_candidates = sorted(
            self.candidates.values(),
            key=lambda c: getattr(c, by),
            reverse=True,
        )
        return sorted_candidates[:n]
    
    def get_centroid_embedding(self) -> np.ndarray:
        """
        Calcula el embedding centroide de la población.
        
        Returns:
            Embedding promedio normalizado
        """
        if not self.candidates:
            return np.zeros(self._embedding_provider.dim)
        
        embeddings = np.array([c.embedding for c in self.candidates.values()])
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        
        if norm > 0:
            centroid = centroid / norm
        
        return centroid

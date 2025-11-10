"""
Text embedding generator for vector database.
"""

from __future__ import annotations

from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

from utils.logger_util import get_logger

logger = get_logger("core.embedder")


class Embedder:
    """
    Generates text embeddings for vector search.
    Uses sentence-transformers for local embedding generation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings

    def load(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return
        
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model loaded")

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if self._model is None:
            self.load()
        
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimension
        
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        return self._dimension


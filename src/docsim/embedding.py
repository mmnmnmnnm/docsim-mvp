from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """
    Lazy-load and return the global SentenceTransformer model instance.

    The model is instantiated only once per process. Subsequent calls reuse
    the same model instance, which is important for performance in services.
    """
    global _model
    if _model is None:
        model = SentenceTransformer(_MODEL_NAME)
        # Increasing the max_seq_length
        # Default 128 token but I raised to 256
        model.max_seq_length = 256
        _model = model
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Vector embedding
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return np.asarray(embedding).ravel()

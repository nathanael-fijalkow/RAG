from __future__ import annotations

from typing import List
import numpy as np
from huggingface_hub import InferenceClient

from .config import get_settings



class Embeddings:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._settings = get_settings()

    def embed(self, texts: List[str]) -> np.ndarray:
        api_key = self._settings.hf_api_key
        if not api_key:
            raise RuntimeError("HUGGINGFACE_API_KEY not set for HF embeddings")
        if len(texts) == 1:
            print(f"[Embeddings] Requesting HF embeddings for one text...")
        else:
            print(f"[Embeddings] Requesting HF embeddings for {len(texts)} texts...")
        client = InferenceClient(
            model=self._model_name,
        )
        embeddings = client.feature_extraction(
            text=texts,
            model=self._model_name,
            normalize=True 
        )
        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        if len(texts) == 1:
            print(f"[Embeddings] HF embedding success for one text.")
        else:
            print(f"[Embeddings] HF embedding success for {len(texts)} texts.")
        return arr

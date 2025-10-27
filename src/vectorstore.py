from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions


class ChromaStore:
    def __init__(self, persist_directory: Optional[str] = None, collection: str = "docs") -> None:
        # Telemetry is disabled via CHROMADB_TELEMETRY env var
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection)

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        import numpy as np
        arr = np.array(embeddings, dtype=np.float32)
        # ChromaDB v1.x requires metadata values to be primitive types (str, int, float, bool)
        safe_metadatas = [{k: str(v) for k, v in md.items()} for md in metadatas]
        self.collection.add(documents=texts, embeddings=arr, metadatas=safe_metadatas, ids=ids)

    def query(self, query_embeddings: List[List[float]], top_k: int = 5, where: Optional[Dict] = None) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        import numpy as np
        arr = np.array(query_embeddings, dtype=np.float32)
        res = self.collection.query(query_embeddings=arr, n_results=top_k, where=where)
        docs = (res.get("documents") or [[]])[0] if res else []
        metas_raw = (res.get("metadatas") or [[]])[0] if res else []
        dists = (res.get("distances") or [[]])[0] if res else []
        metas = [dict(md) for md in metas_raw] if metas_raw else []
        return docs, metas, dists

    def get_by_where(self, where: Dict, limit: int = 10) -> Tuple[List[str], List[Dict[str, Any]]]:
        res = self.collection.get(where=where)
        docs = (res.get("documents") or [])[:limit] if res else []
        metas_raw = (res.get("metadatas") or [])[:limit] if res else []
        metas = [dict(md) for md in metas_raw] if metas_raw else []
        return docs, metas

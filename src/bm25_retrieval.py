from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path


class BM25Retriever:
    """BM25 retrieval using rank_bm25 library."""
    
    def __init__(self, persist_directory: Optional[str] = None, collection: str = "docs"):
        """
        Initialize BM25 retriever.
        
        Args:
            persist_directory: Directory to save/load BM25 index
            collection: Collection name (used for file naming)
        """
        self.persist_directory = persist_directory
        self.collection = collection
        self.corpus: List[str] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        self.bm25 = None
        self._load_index()
    
    def _get_index_path(self) -> Path:
        """Get path to BM25 index file."""
        if not self.persist_directory:
            raise ValueError("persist_directory must be set to save/load BM25 index")
        base_path = Path(self.persist_directory)
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / f"bm25_{self.collection}.pkl"
    
    def _load_index(self):
        """Load BM25 index from disk if it exists."""
        try:
            index_path = self._get_index_path()
            if index_path.exists():
                with open(index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.corpus = data['corpus']
                    self.metadatas = data['metadatas']
                    self.ids = data['ids']
                    self.bm25 = data['bm25']
                # print(f"[BM25] Loaded index with {len(self.corpus)} documents from {index_path}")
        except Exception as e:
            print(f"[BM25] Could not load index: {e}")
    
    def _save_index(self):
        """Save BM25 index to disk."""
        try:
            index_path = self._get_index_path()
            with open(index_path, 'wb') as f:
                pickle.dump({
                    'corpus': self.corpus,
                    'metadatas': self.metadatas,
                    'ids': self.ids,
                    'bm25': self.bm25
                }, f)
            print(f"[BM25] Saved index with {len(self.corpus)} documents to {index_path}")
        except Exception as e:
            print(f"[BM25] Could not save index: {e}")
    
    def add_texts(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Add documents to BM25 index.
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
            ids: List of document IDs
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")
        
        self.corpus.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Tokenize corpus (simple whitespace tokenization)
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"[BM25] Indexed {len(texts)} new documents (total: {len(self.corpus)})")
        self._save_index()
    
    def query(self, query: str, top_k: int = 5, where: Optional[Dict] = None) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Query BM25 index.
        
        Args:
            query: Query string
            top_k: Number of results to return
            where: Optional metadata filter (currently not implemented for BM25)
        
        Returns:
            Tuple of (documents, metadatas, scores)
        """
        if not self.bm25 or not self.corpus:
            print("[BM25] Index is empty")
            return [], [], []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Apply metadata filter if provided
        if where:
            filtered_indices = []
            for idx in top_indices:
                meta = self.metadatas[idx]
                # Simple equality filter
                match = all(meta.get(k) == str(v) for k, v in where.items())
                if match:
                    filtered_indices.append(idx)
            top_indices = filtered_indices[:top_k]
        
        # Return results
        docs = [self.corpus[i] for i in top_indices]
        metas = [self.metadatas[i] for i in top_indices]
        result_scores = [float(scores[i]) for i in top_indices]
        
        return docs, metas, result_scores
    
    def count(self) -> int:
        """Return number of documents in index."""
        return len(self.corpus)

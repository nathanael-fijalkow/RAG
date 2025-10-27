from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from rich import print

import numpy as np

from .loaders import Document
from .chunking import chunk_fixed, chunk_recursive, chunk_semantic, Chunk
from .embeddings import Embeddings
from .vectorstore import ChromaStore
from .llm import LLM
from .retrieval import mmr, rerank
from .bm25_retrieval import BM25Retriever
from .merge import merge_results


Chunker = Callable[[str, Dict[str, str]], List[Chunk]]


def get_chunker(name: str) -> Chunker:
    name = name.lower()
    if name == "fixed":
        return lambda text, md: chunk_fixed(text, md, size=600, overlap=100)
    if name == "recursive":
        return lambda text, md: chunk_recursive(text, md, target=700)
    if name == "semantic":
        return lambda text, md: chunk_semantic(text, md, use_llm_summary=True, max_chunk_size=2000)
    raise ValueError("Unknown chunking strategy")


@dataclass
class IndexStats:
    docs: int
    chunks: int


class Indexer:
    def __init__(self, persist: Optional[str] = None, collection: str = "docs", emb_model: str = "BAAI/bge-small-en-v1.5", use_bm25: bool = True):
        self.emb = Embeddings(emb_model)
        self.store = ChromaStore(persist_directory=persist, collection=collection)
        self.use_bm25 = use_bm25
        self.bm25_store = BM25Retriever(persist_directory=persist, collection=collection) if use_bm25 else None

    def index_documents(self, docs: List[Document], chunking: str = "fixed") -> IndexStats:
        chunker = get_chunker(chunking)
        texts: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []
        for i, doc in enumerate(docs):
            print(f"[Indexer] Chunking document {i+1}/{len(docs)}: {doc.metadata.get('source','?')} ({len(doc.content)} chars)")
            chunks = chunker(doc.content, doc.metadata)
            for j, ch in enumerate(chunks):
                # print(f"  - Chunk {j+1}/{len(chunks)}: {len(ch.text)} chars"    )
                texts.append(ch.text)
                md = {k: str(v) for k, v in ch.metadata.items()}
                md["chunk_index"] = str(j)
                metadatas.append(md)
                ids.append(f"{md.get('doc_id','doc')}-{md.get('page','0')}-{j}")
        if not texts:
            return IndexStats(docs=len(docs), chunks=0)
        vecs = self.emb.embed(texts)
        self.store.add_texts(texts, vecs.tolist(), metadatas, ids)
        
        # Also index with BM25 if enabled
        if self.use_bm25 and self.bm25_store:
            self.bm25_store.add_texts(texts, metadatas, ids)
        
        return IndexStats(docs=len(docs), chunks=len(texts))

class RAG:
    def __init__(self, persist: Optional[str] = None, collection: str = "docs", emb_model: str = "BAAI/bge-small-en-v1.5", use_bm25: bool = True):
        self.emb = Embeddings(emb_model)
        self.store = ChromaStore(persist_directory=persist, collection=collection)
        self.llm = LLM()
        self.use_bm25 = use_bm25
        self.bm25_store = BM25Retriever(persist_directory=persist, collection=collection) if use_bm25 else None

    def retrieve(self, question: str, top_k: int = 5, where: Optional[Dict] = None, mmr_lambda: float = 0.6, use_mmr: bool = True, use_rerank: bool = False, retrieval_mode: str = "hybrid"):
        """
        Retrieve relevant documents.
        
        Args:
            question: Query string
            top_k: Number of results to return
            where: Metadata filter
            mmr_lambda: MMR diversity parameter (0=max diversity, 1=max relevance)
            use_mmr: Whether to use MMR for diversity
            use_rerank: Whether to use LLM reranking
            retrieval_mode: "vector" (embedding only), "bm25" (keyword only), or "hybrid" (both merged)
        
        Returns:
            Tuple of (documents, metadatas)
        """
        # Handle different retrieval modes
        if retrieval_mode == "bm25":
            # BM25 only
            if not self.bm25_store:
                print("[RAG] Warning: BM25 not available, falling back to vector search")
                retrieval_mode = "vector"
            else:
                print(f"[RAG] Using BM25 retrieval")
                docs, metas, scores = self.bm25_store.query(question, top_k=top_k, where=where)
                if not docs:
                    return [], []
        
        elif retrieval_mode == "hybrid":
            # Hybrid: merge vector and BM25 results
            if not self.bm25_store:
                print("[RAG] Warning: BM25 not available, falling back to vector search")
                retrieval_mode = "vector"
            else:
                print(f"[RAG] Using hybrid retrieval (vector + BM25)")
                # Get vector results
                qv = self.emb.embed([question])[0]
                vector_docs, vector_metas, vector_dists = self.store.query([qv.tolist()], top_k=top_k * 2, where=where)
                # Convert distances to scores (lower distance = higher score)
                vector_scores = [1.0 / (1.0 + d) for d in vector_dists]
                
                # Get BM25 results
                bm25_docs, bm25_metas, bm25_scores = self.bm25_store.query(question, top_k=top_k * 2, where=where)
                
                # Merge results using RRF
                docs, metas = merge_results(
                    [(vector_docs, vector_metas, vector_scores),
                     (bm25_docs, bm25_metas, bm25_scores)],
                    method="rrf",
                    top_k=max(top_k * 2, 8)  # Overfetch for MMR/rerank
                )
                if not docs:
                    return [], []
        
        else:  # vector mode (default)
            print(f"[RAG] Using vector retrieval")
            qv = self.emb.embed([question])[0]
            # initial overfetch
            docs, metas, dists = self.store.query([qv.tolist()], top_k=max(top_k * 4, 8), where=where)
            if not docs:
                return [], []
        
        # MMR selection (only for vector/hybrid modes)
        if use_mmr and retrieval_mode != "bm25":
            # embed candidates for MMR diversity
            cands = self.emb.embed(docs)
            qv = self.emb.embed([question])[0]  # Re-embed query if in hybrid mode
            order = mmr(qv, cands, lambda_=mmr_lambda, top_k=top_k)
            docs = [docs[i] for i in order]
            metas = [metas[i] for i in order]
        else:
            docs = docs[:top_k]
            metas = metas[:top_k]
        
        # Optional reranking
        if use_rerank:
            print("\n\nThe original order before reranking:")
            for i, md in enumerate(metas, 1):
                doc_text = docs[i - 1]
                excerpt = (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
                print(f"[{i}] {md.get('source')}#page={md.get('page')}\n> {excerpt}")
            order2 = rerank(question, docs)
            docs = [docs[i] for i in order2[:top_k]]
            metas = [metas[i] for i in order2[:top_k]]
            print("\n\nAfter reranking:")
            for i, md in enumerate(metas, 1):
                doc_text = docs[i - 1]
                excerpt = (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
                print(f"[{i}] {md.get('source')}#page={md.get('page')}\n> {excerpt}")
        return docs, metas

    def answer(self, question: str, context_docs: List[str], metadatas: List[Dict]) -> str:
        context = "\n\n".join(f"[Source {i+1}] {md.get('source','?')} p.{md.get('page','?')}\n{doc}" for i, (doc, md) in enumerate(zip(context_docs, metadatas)))
        system = "You are a helpful assistant. Answer using only the provided context. Cite sources inline like [Source 1]. If unsure, say you don't know."
        prompt = f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer concisely with sources."
        return self.llm.generate(prompt, system=system)

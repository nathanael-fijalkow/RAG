from __future__ import annotations

from typing import Dict, List, Tuple


def merge_results(
    results_list: List[Tuple[List[str], List[Dict], List[float]]],
    method: str = "rrf",
    top_k: int = 5,
    rrf_k: int = 60
) -> Tuple[List[str], List[Dict]]:
    """
    Merge results from multiple retrievers.
    
    Args:
        results_list: List of (docs, metadatas, scores) tuples from different retrievers
        method: Merge method - "rrf" (Reciprocal Rank Fusion) or "score" (score-based)
        top_k: Number of final results to return
        rrf_k: RRF constant (default 60)
    
    Returns:
        Tuple of (merged_docs, merged_metadatas)
    """
    if not results_list:
        return [], []
    
    # Build a mapping from document text to its metadata and scores
    doc_to_info: Dict[str, Dict] = {}
    
    for retriever_idx, (docs, metas, scores) in enumerate(results_list):
        for rank, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
            doc_key = doc  # Use document text as key
            
            if doc_key not in doc_to_info:
                doc_to_info[doc_key] = {
                    'doc': doc,
                    'meta': meta,
                    'scores': [],
                    'ranks': [],
                    'retrievers': []
                }
            
            doc_to_info[doc_key]['scores'].append(score)
            doc_to_info[doc_key]['ranks'].append(rank)
            doc_to_info[doc_key]['retrievers'].append(retriever_idx)
    
    # Calculate final scores based on method
    final_scores = []
    
    if method == "rrf":
        # Reciprocal Rank Fusion: 1 / (k + rank)
        for doc_key, info in doc_to_info.items():
            rrf_score = sum(1.0 / (rrf_k + rank) for rank in info['ranks'])
            final_scores.append((rrf_score, doc_key))
    
    elif method == "score":
        # Simple score fusion: sum of normalized scores
        # First normalize scores per retriever
        max_scores = [max([s for _, _, scores in results_list for s in scores] or [1.0]) 
                     for _ in results_list]
        
        for doc_key, info in doc_to_info.items():
            normalized_score = sum(
                score / max(max_scores[ret_idx], 1e-8)
                for score, ret_idx in zip(info['scores'], info['retrievers'])
            )
            final_scores.append((normalized_score, doc_key))
    
    else:
        raise ValueError(f"Unknown merge method: {method}")
    
    # Sort by final score and take top_k
    final_scores.sort(reverse=True)
    top_docs = []
    top_metas = []
    
    for score, doc_key in final_scores[:top_k]:
        info = doc_to_info[doc_key]
        top_docs.append(info['doc'])
        top_metas.append(info['meta'])
    
    print(f"[Merge] Merged {len(results_list)} result sets using {method}, returning {len(top_docs)} documents")
    return top_docs, top_metas

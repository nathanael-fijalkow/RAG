from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import json
import re

from .llm import LLM


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def mmr(query: np.ndarray, candidates: np.ndarray, lambda_: float = 0.5, top_k: int = 5) -> List[int]:
    selected: List[int] = []
    idxs = list(range(len(candidates)))
    sim_query = np.array([cosine_sim(query, c) for c in candidates])
    while len(selected) < min(top_k, len(idxs)):
        mmr_scores = []
        for i in idxs:
            if not selected:
                diversity = 0.0
            else:
                diversity = max(cosine_sim(candidates[i], candidates[j]) for j in selected)
            score = lambda_ * sim_query[i] - (1 - lambda_) * diversity
            mmr_scores.append((score, i))
        mmr_scores.sort(reverse=True)
        best = mmr_scores[0][1]
        selected.append(best)
        idxs.remove(best)
    return selected


def rerank(query: str, passages: List[str]) -> List[int]:
    """
    Rerank passages based on their relevance to the query using an LLM.
    Returns a list of indices (0-based) sorted by relevance.
    """
    print(f"\n\n[Reranker] Requesting LLM for reranking...")

    prompt = f"**Query**: {query}\n"
    for i, doc in enumerate(passages):
        prompt += f"**Passage {i}**: {doc}\n"
    
    system = (
        "You are a helpful assistant that ranks passages based on their relevance to the query. "
        "Return a JSON list of indices (0-based) that sorts the passages from most to least relevant. "
        "Include all passages. Example output: [2, 0, 1, 3]"
    )

    llm = LLM()
    text = llm.generate(prompt, system=system)
    print(f"[Reranker] LLM response received: {text}")

    # Parse the output to extract indices
    # Try to parse as JSON first
    try:
        indices = json.loads(text)
        if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
            print(f"[Reranker] Parsed {len(indices)} indices from JSON")
            return indices
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try to extract numbers from the text
    numbers = re.findall(r'\d+', text)
    if numbers:
        indices = [int(n) for n in numbers]
        # Filter out indices that are out of range
        indices = [i for i in indices if 0 <= i < len(passages)]
        print(f"[Reranker] Extracted {len(indices)} indices from text")
        return indices
    
    # Fallback: return original order
    print(f"[Reranker] Warning: Could not parse indices, returning original order")
    return list(range(len(passages)))

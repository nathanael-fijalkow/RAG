from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .pipeline import RAG
from .llm import LLM


def hyde_query(llm: LLM, question: str) -> str:
    prompt = f"Generate a plausible answer to the question to use as a retrieval query.\nQuestion: {question}\nAnswer:"
    return llm.generate(prompt, system="You generate concise, factual drafts for retrieval use.")

def parse_filter(filter_str: Optional[str]) -> Optional[Dict]:
    if not filter_str:
        return None
    # simple key=value;key2=value2 parser
    where: Dict[str, str] = {}
    for part in filter_str.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            where[k.strip()] = v.strip()
    return where or None

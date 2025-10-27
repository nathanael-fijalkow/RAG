from __future__ import annotations

import argparse
from pathlib import Path
from rich import print
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from langgraph.graph import StateGraph, START, END

from typing import Dict, Any

from src.pipeline import RAG
from src.tools import web_search
from src.llm import LLM

def decide(state: Dict[str, Any]) -> str:
    # return "web"
    q: str = state["question"]
    llm: LLM = state["rag"].llm
    prompt = f"Determine whether the query should be answered by a web search, synthesis, or retrieval.\nQuestion: {q}\nAnswer:"
    answer = llm.generate(prompt, system="You decide the best method to answer the question: 'web' (for recent pieces of information), 'synth' (for summarising a single document), or 'retrieve' (to obtain a single fact).").strip().lower()
    print(f"Decided method: {answer}\n\n")
    return answer

def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    rag: RAG = state["rag"]
    docs, metas = rag.retrieve(state["question"], top_k=6, use_mmr=True)
    answer = rag.answer(state["question"], docs, metas)
    return {**state, "answer": answer, "sources": metas, "docs": docs}


def node_synth(state: Dict[str, Any]) -> Dict[str, Any]:
    rag: RAG = state["rag"]
    # 1) Retrieve the top-k most relevant chunks for the user's question
    chunk_docs, chunk_metas = rag.retrieve(state["question"], top_k=5, use_mmr=False, use_rerank=False)

    if not chunk_docs:
        return {**state, "answer": "", "sources": [], "docs": []}

    # 2) Collect up to 5 unique documents (full) that contain these chunks
    #    We identify documents by their doc_id metadata and fetch all chunks for each doc
    unique_doc_ids = []
    for md in chunk_metas:
        did = md.get("doc_id")
        if did and did not in unique_doc_ids:
            unique_doc_ids.append(did)
        if len(unique_doc_ids) >= 5:
            break

    full_docs: list[str] = []
    full_metas: list[Dict[str, Any]] = []

    for did in unique_doc_ids:
        # Fetch all chunks for this document using metadata filter
        all_chunks_texts, all_chunks_metas = rag.store.get_by_where({"doc_id": did}, limit=10_000)
        if not all_chunks_texts:
            # Fallback: if nothing fetched, skip
            continue
        # Order chunks by their chunk_index if available
        paired = list(zip(all_chunks_texts, all_chunks_metas))
        def _idx(md: Dict[str, Any]) -> int:
            try:
                return int(str(md.get("chunk_index", "0")).split("-")[0])
            except Exception:
                return 0
        paired.sort(key=lambda x: _idx(x[1]))

        merged_text = "\n\n".join(t for t, _ in paired)
        # Build a representative metadata record for the full document
        rep_meta = dict(paired[0][1])
        rep_meta["page"] = "all"
        rep_meta["source"] = rep_meta.get("source", did)
        rep_meta["doc_id"] = did
        full_docs.append(merged_text)
        full_metas.append(rep_meta)

    # If we couldn't assemble full docs (unlikely), fallback to the original chunks
    if not full_docs:
        full_docs, full_metas = chunk_docs, chunk_metas

    # 3) Synthesize the answer using the full documents' content
    answer = rag.answer(state["question"], full_docs, full_metas)
    return {**state, "answer": answer, "sources": full_metas, "docs": full_docs}


def node_web(state: Dict[str, Any]) -> Dict[str, Any]:
    rag: RAG = state["rag"]
    results = web_search(state["question"], max_results=5)
    context = "\n\n".join(f"[{i+1}] {r['title']}\n{r['body']}\n{r['href']}" for i, r in enumerate(results))
    prompt = f"Use the web results below to answer. Cite sources like [1].\nQuestion: {state['question']}\n\nWeb Results:\n{context}"
    answer = rag.llm.generate(prompt, system="You answer using provided snippets and cite sources.")
    metas = [{"source": r["href"], "page": "web"} for r in results]
    return {**state, "answer": answer, "sources": metas, "docs": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--persist", default=str(Path(__file__).resolve().parents[1] / ".chroma"))
    args = ap.parse_args()

    rag = RAG(persist=args.persist)

    sg = StateGraph(dict)
    sg.add_node("retrieve", node_retrieve)
    sg.add_node("synth", node_synth)
    sg.add_node("web", node_web)

    # Use conditional_edges to route from START based on the decide function
    sg.add_conditional_edges(
        START,
        decide,
        {
            "retrieve": "retrieve",
            "synth": "synth",
            "web": "web"
        }
    )
    sg.add_edge("retrieve", END)
    sg.add_edge("synth", END)
    sg.add_edge("web", END)
    graph = sg.compile()

    out = graph.invoke({"question": args.question, "rag": rag})
    print("\n[bold]Answer[/bold]\n")
    print(out.get("answer", ""))
    print("\n[bold]Sources[/bold]")
    for i, md in enumerate(out.get("sources", []), 1):
        doc_text = out.get("docs", [])[i - 1]
        excerpt = (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
        print(f"[{i}] {md.get('source')}#page={md.get('page')}\n> {excerpt}")

if __name__ == "__main__":
    main()

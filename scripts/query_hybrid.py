from __future__ import annotations

import argparse
from pathlib import Path
from rich import print
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import RAG


def main():
    ap = argparse.ArgumentParser(description='Query with different retrieval modes')
    ap.add_argument("--question", required=True, help="Question to ask")
    ap.add_argument("--top_k", type=int, default=4, help="Number of results")
    ap.add_argument("--persist", default=str(Path(__file__).resolve().parents[1] / ".chroma"), help="Chroma persist directory")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid", 
                    help="Retrieval mode: vector (embeddings only), bm25 (keyword only), or hybrid (both merged)")
    ap.add_argument("--dry-run", action="store_true", help="Only show retrieved chunks and sources, skip LLM call")
    args = ap.parse_args()

    rag = RAG(persist=args.persist, use_bm25=True)
    
    print(f"\n[bold cyan]Retrieval Mode:[/bold cyan] {args.mode}")
    
    docs, metas = rag.retrieve(
        args.question, 
        top_k=args.top_k, 
        use_mmr=True, 
        use_rerank=False,
        retrieval_mode=args.mode
    )
    
    if not docs:
        print("[red]No results found. Did you index documents first?[/red]")
        return
    
    if not args.dry_run:
        answer = rag.answer(args.question, docs, metas)
        print("\n[bold]Answer[/bold]\n")
        print(answer)
    else:
        print("\n[bold]Dry run[/bold]: showing retrieved excerpts only.")
    
    print("\n[bold]Sources[/bold]")
    for i, (doc, md) in enumerate(zip(docs, metas), 1):
        excerpt = (doc[:200] + "...") if len(doc) > 200 else doc
        print(f"[{i}] {md.get('source')}#page={md.get('page')}\n> {excerpt}\n")


if __name__ == "__main__":
    main()

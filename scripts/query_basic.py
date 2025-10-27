from __future__ import annotations

import argparse
from pathlib import Path
from rich import print
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import RAG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--persist", default=str(Path(__file__).resolve().parents[1] / ".chroma"))
    ap.add_argument("--dry-run", action="store_true", help="Only show retrieved chunks and sources, skip LLM call")
    args = ap.parse_args()

    rag = RAG(persist=args.persist)
    docs, metas = rag.retrieve(args.question, top_k=args.top_k, use_mmr=True, use_rerank=False)
    if not docs:
        print("No results. Did you index first?")
        return
    if not args.dry_run:
        answer = rag.answer(args.question, docs, metas)
        print("\n[bold]Answer[/bold]\n")
        print(answer)
    else:
        print("\n[bold]Dry run[/bold]: showing retrieved excerpts only.")
    print("\n[bold]Sources[/bold]")
    for i, md in enumerate(metas, 1):
        doc_text = docs[i - 1]
        excerpt = (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
        print(f"[{i}] {md.get('source')}#page={md.get('page')}\n> {excerpt}")


if __name__ == "__main__":
    main()

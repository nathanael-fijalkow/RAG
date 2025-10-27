from __future__ import annotations

import argparse
from pathlib import Path
from rich import print
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import RAG
from src.enhanced import hyde_query, parse_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--persist", default=str(Path(__file__).resolve().parents[1] / ".chroma"))
    ap.add_argument("--hyde", action="store_true")
    ap.add_argument("--mmr", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--filter", default=None, help="key=value;key2=value2")
    args = ap.parse_args()

    rag = RAG(persist=args.persist)
    if args.hyde:
        q = hyde_query(rag.llm, args.question)
        print(f"The HyDE plausible answer: {q}\n\n")
    else:
        q = args.question
    where = parse_filter(args.filter)
    docs, metas = rag.retrieve(q, top_k=6, where=where, use_mmr=args.mmr, use_rerank=args.rerank)
    if not docs:
        print("No results. Did you index first?")
        return
    answer = rag.answer(args.question, docs[:6], metas[:6])
    print("\n[bold]Answer[/bold]\n")
    print(answer)
    print("\n[bold]Sources[/bold]")
    for i, md in enumerate(metas, 1):
        doc_text = docs[i - 1]
        excerpt = (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
        print(f"[{i}] {md.get('source')}#page={md.get('page')}\n> {excerpt}")


if __name__ == "__main__":
    main()

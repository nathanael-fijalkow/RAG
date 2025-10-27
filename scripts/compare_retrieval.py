from __future__ import annotations

import argparse
from pathlib import Path
from rich import print
from rich.table import Table
from rich.console import Console
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import RAG


def main():
    ap = argparse.ArgumentParser(description='Compare different retrieval modes')
    ap.add_argument("--question", required=True, help="Question to ask")
    ap.add_argument("--top_k", type=int, default=3, help="Number of results per mode")
    ap.add_argument("--persist", default=str(Path(__file__).resolve().parents[1] / ".chroma"), 
                    help="Chroma persist directory")
    args = ap.parse_args()

    console = Console()
    rag = RAG(persist=args.persist, use_bm25=True)
    
    console.print(f"\n[bold cyan]Question:[/bold cyan] {args.question}\n")
    
    modes = ["vector", "bm25", "hybrid"]
    results = {}
    
    for mode in modes:
        console.print(f"[yellow]Running {mode} retrieval...[/yellow]")
        docs, metas = rag.retrieve(
            args.question, 
            top_k=args.top_k, 
            use_mmr=False,  # Disable MMR for fair comparison
            use_rerank=False,
            retrieval_mode=mode
        )
        results[mode] = (docs, metas)
    
    # Create comparison table
    table = Table(title=f"Retrieval Comparison (Top {args.top_k})", show_header=True)
    table.add_column("Rank", style="cyan", width=5)
    table.add_column("Vector", style="green")
    table.add_column("BM25", style="yellow")
    table.add_column("Hybrid", style="magenta")
    
    for i in range(args.top_k):
        row = [f"{i+1}"]
        for mode in modes:
            docs, metas = results[mode]
            if i < len(docs):
                doc = docs[i]
                meta = metas[i]
                excerpt = (doc[:100] + "...") if len(doc) > 100 else doc
                source = Path(meta.get('source', 'unknown')).stem
                cell_text = f"[bold]{source}[/bold]\n{excerpt}"
            else:
                cell_text = "[dim]No result[/dim]"
            row.append(cell_text)
        table.add_row(*row)
    
    console.print("\n")
    console.print(table)
    
    # Show overlap statistics
    console.print("\n[bold cyan]Overlap Analysis:[/bold cyan]")
    
    def get_doc_ids(mode):
        docs, metas = results[mode]
        return set(Path(m.get('source', '')).stem + f"_{i}" for i, m in enumerate(metas))
    
    vector_ids = get_doc_ids("vector")
    bm25_ids = get_doc_ids("bm25")
    hybrid_ids = get_doc_ids("hybrid")
    
    vector_bm25_overlap = len(vector_ids & bm25_ids)
    console.print(f"  Vector ∩ BM25: {vector_bm25_overlap}/{args.top_k} documents")
    console.print(f"  Unique to Vector: {len(vector_ids - bm25_ids)} documents")
    console.print(f"  Unique to BM25: {len(bm25_ids - vector_ids)} documents")
    console.print(f"  Hybrid combines both with {len(hybrid_ids)} results")


if __name__ == "__main__":
    main()

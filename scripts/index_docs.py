from __future__ import annotations

import argparse
from pathlib import Path
from rich import print
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.loaders import load_folder
from src.pipeline import Indexer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(Path(__file__).resolve().parents[1] / "data" / "docs" / "wiki_finewiki_en"))
    ap.add_argument("--chunking", default="fixed", choices=["fixed", "recursive", "semantic"])
    ap.add_argument("--persist", default=str(Path(__file__).resolve().parents[1] / ".chroma"))
    args = ap.parse_args()

    docs = load_folder(args.folder)
    indexer = Indexer(persist=args.persist)
    stats = indexer.index_documents(docs, chunking=args.chunking)
    print({"indexed_docs": stats.docs, "indexed_chunks": stats.chunks, "persist": args.persist})


if __name__ == "__main__":
    main()

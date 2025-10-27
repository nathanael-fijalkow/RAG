"""
Inspect Chroma vector database: print statistics and sample documents.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vectorstore import ChromaStore


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def inspect_database(persist_dir: str, collection_name: str = "rag_collection", sample_size: int = 3):
    """
    Inspect the Chroma database and print statistics.
    
    Args:
        persist_dir: Path to Chroma persistence directory
        collection_name: Name of the collection to inspect
        sample_size: Number of sample documents to display
    """
    print(f"📊 Inspecting Chroma Database")
    print_separator()
    print(f"Location: {persist_dir}")
    print(f"Collection: {collection_name}")
    print()
    
    # Initialize vector store
    try:
        vectorstore = ChromaStore(
            persist_directory=persist_dir,
            collection=collection_name
        )
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return
    
    # Get collection stats
    try:
        collection = vectorstore.collection
        count = collection.count()
        
        print(f"📈 Collection Statistics")
        print_separator("-")
        print(f"Total documents: {count:,}")
        print()
        
        if count == 0:
            print("⚠️  Database is empty. Run indexing first.")
            return
        
        # Get all documents with metadata
        results = collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        
        # Analyze metadata
        if results and results.get("metadatas"):
            metadatas = results["metadatas"]
            
            # Collect unique values for each metadata key
            metadata_keys = {}
            for meta in metadatas:
                if meta:
                    for key, value in meta.items():
                        if key not in metadata_keys:
                            metadata_keys[key] = set()
                        metadata_keys[key].add(str(value))
            
            print(f"📋 Metadata Fields")
            print_separator("-")
            for key, values in sorted(metadata_keys.items()):
                print(f"  {key}: {len(values)} unique value(s)")
                if len(values) <= 10:
                    for val in sorted(values):
                        print(f"    - {val}")
                else:
                    for val in sorted(list(values)[:5]):
                        print(f"    - {val}")
                    print(f"    ... and {len(values) - 5} more")
            print()
        
        # Embedding dimension
        if results.get("embeddings") is not None:
            embeddings = results["embeddings"]
            if embeddings is not None and len(embeddings) > 0:
                embedding_dim = len(embeddings[0])
                print(f"🔢 Embedding Dimension: {embedding_dim}")
                print()
        
        # Sample documents
        print(f"📄 Sample Documents (showing {min(sample_size, count)})")
        print_separator("-")
        
        sample_results = collection.get(
            limit=sample_size,
            include=["documents", "metadatas"]
        )
        
        if sample_results.get("documents") is not None:
            docs = sample_results["documents"]
            metas = sample_results.get("metadatas", [])
            ids = sample_results.get("ids", [])
            
            for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas), 1):
                print(f"\n[Document {i}]")
                print(f"ID: {doc_id}")
                if meta:
                    print("Metadata:")
                    for key, value in sorted(meta.items()):
                        print(f"  {key}: {value}")
                
                # Show first 200 characters of document
                doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"Content: {doc_preview}")
                print_separator("-", 40)
        
        print()
        print("✅ Inspection complete")
        
    except Exception as e:
        print(f"❌ Error inspecting database: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Inspect Chroma database')
    parser.add_argument('--persist', type=str, default='.chroma',
                        help='Path to Chroma persist directory (default: .chroma)')
    parser.add_argument('--collection', type=str, default='docs',
                        help='Name of the collection to inspect (default: docs)')
    parser.add_argument('--sample', type=int, default=3,
                        help='Number of sample documents to display (default: 3)')
    
    args = parser.parse_args()
    
    # Check if persist directory exists
    persist_path = Path(args.persist)
    if not persist_path.exists():
        print(f"❌ Error: Database directory '{args.persist}' does not exist.")
        print("   Run indexing first with: python scripts/index_pdfs.py --persist .chroma")
        sys.exit(1)
    
    inspect_database(args.persist, args.collection, args.sample)


if __name__ == "__main__":
    main()

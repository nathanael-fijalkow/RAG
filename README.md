# RAG From Scratch

This codebase provides clean, educational implementations of modern Retrieval-Augmented Generation (RAG) techniques with simple code and demos. Includes:

- Classic RAG foundations and improved RAG (chunking, retrieval, reranking, filtering)
- Agentic RAG with decision-making, adaptive steps, and web search

Uses either Gemini API or Hugging Face Inference API for the LLM. Embeddings run locally for speed.

## Quick Start

1. Install dependencies (Python 3.10+ recommended)

2. Set your API of choice in `.env`:

- For Gemini:
  - `LLM_PROVIDER=gemini`
  - `GEMINI_API_KEY=...`
  - `GEMINI_MODEL=gemini-1.5-pro` (default)

- For Hugging Face Inference API:
  - `LLM_PROVIDER=huggingface`
  - `HUGGINGFACE_API_KEY=...`
  - `HF_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct` (or another available model)

3. Prepare data

Place documents into `data/docs/`

You can use the following to extract Wikipedia articles of a curated list of famous physicists:

```bash
python scripts/extraction_wikipedia.py
```

Another option, using the Hugging Face dataset FineWiki:

```bash
python scripts/download_wiki_dataset.py
```

**Note:** The FineWiki dataset is highly structured and generally higher quality for Wikipedia-like content.

4. Chunk and Index Documents

Three chunking strategies available:

### Fixed Chunking (default)

- Splits text into fixed-size chunks (600 chars) with overlap (100 chars)
- Fast and predictable, good for general use

```bash
python scripts/index_docs.py --chunking fixed --persist .chroma
```

### Recursive Chunking

- Splits by paragraphs, then sentences if needed
- Target size: 700 chars
- Better preserves semantic boundaries

```bash
python scripts/index_docs.py --chunking recursive --persist .chroma
```

### Semantic Chunking

- Splits based on document structure using markdown headers or Wikipedia format
- Supports: `# H1`, `## H2`, `=== title ===` (major sections)
- And: `### H3`, `#### H4`, `==== subtitle ====` (subsections)
- Automatically removes HTML/Markdown comments before processing
- Adds context headers (filename, section, subsection) to each chunk
- Optional LLM-generated summaries for each section
- Best for structured documents (Wikipedia, FineWiki, technical docs)

```bash
python scripts/index_docs.py --chunking semantic --persist .chroma
```

**Note:** Indexing automatically creates both vector (Chroma) and BM25 (keyword) indices for hybrid retrieval.

To inspect the Chroma database:

```bash
python scripts/inspect_chroma.py
# Or with custom options:
python scripts/inspect_chroma.py --persist .chroma --collection docs --sample 5
```

5. Basic RAG

```bash
python scripts/query_basic.py --question "Where was Albert Einstein born?" --top_k 4 --persist .chroma
# Add --dry-run to see retrieved chunks without LLM call
```

6. Query with Different Retrieval Modes

### Hybrid Retrieval (Recommended)

Combines vector (semantic) and BM25 (keyword) search using Reciprocal Rank Fusion:

```bash
python scripts/query_hybrid.py --question "What is Einstein known for?" --mode hybrid
```

### Vector-Only Retrieval

Traditional embedding-based semantic search:

```bash
python scripts/query_hybrid.py --question "What theories did Einstein develop?" --mode vector
```

### BM25-Only Retrieval

Pure keyword-based search (best for exact term matching):

```bash
python scripts/query_hybrid.py --question "quantum mechanics photoelectric effect" --mode bm25
```

### Compare Retrieval Modes

Side-by-side comparison of all three modes:

```bash
python scripts/compare_retrieval.py --question "What is relativity?" --top_k 3
```

7. Enhanced RAG (HyDE, MMR, rerank, metadata filter)

```bash
python scripts/query_enhanced.py --question "Where was Otto Sackur born?" --persist .chroma --hyde --mmr --filter doc_id=Otto_Sackur;page=1 --rerank
```

8. Agentic RAG with decision graph + choice between web search, summary, and retrieval

```bash
python scripts/query_agentic.py --question "What are the latest findings on RAG evaluation?" --persist .chroma
```

9. Evaluate with RAGAS

```bash
python scripts/eval_ragas.py --questions questions.txt --mode hybrid --top_k 4 --persist .chroma

# Or with references (CSV with columns: question, ground_truth)
python scripts/eval_ragas.py --questions eval_set.csv --mode hybrid --top_k 4 --persist .chroma --rerank
```

Notes:

- questions.txt: one question per line.
- eval_set.csv: must contain a 'question' column and may contain an optional 'ground_truth' column.
- Metrics reported: context_precision, answer_relevancy, faithfulness; adds context_recall and (if available) answer_correctness when ground_truth is provided.

## Modules Covered

### Chunking Strategies

- **Fixed**: Fixed-size chunks (600 chars) with overlap (100 chars)
- **Recursive**: Split by paragraphs/sentences, target 700 chars
- **Semantic**: Structure-aware splitting on markdown headers or Wikipedia format
  - Supports `# H1`, `## H2`, `=== title ===` (major sections)
  - Supports `### H3`, `#### H4`, `==== subtitle ====` (subsections)
  - Removes HTML/Markdown comments automatically
  - Adds context headers (filename, section, subsection)
  - Optional LLM-generated summaries
  - Works with Wikipedia and FineWiki datasets

### Retrieval Methods

- **Vector Search**: Semantic search using HuggingFace embeddings (BAAI/bge-small-en-v1.5)
- **BM25**: Keyword-based search using rank-bm25 library
- **Hybrid**: Combines vector + BM25 using Reciprocal Rank Fusion (RRF)
- **MMR**: Maximal Marginal Relevance for diversity
- **Reranking**: LLM-based reranking via Gemini API

### Advanced Features

- **HyDE**: Hypothetical Document Embeddings
- **Metadata Filtering**: Filter by document ID, page, or custom fields
- **Agentic RAG**: LangGraph-based decision routing
  - Precise retrieval vs document synthesis vs web search
- **Web Search**: DuckDuckGo via the ddgs library (no API key required)

### Data Sources

- Wikipedia page extraction
- Internet Archive (curated public-domain PDFs and texts)
- Custom document upload to `data/docs/`

## Installation

Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your keys.

## Folder Structure

- `src/` core library code for the demos
  - `chunking.py` - Chunking strategies (fixed, recursive, semantic)
  - `vectorstore.py` - ChromaDB vector store wrapper
  - `bm25_retrieval.py` - BM25 keyword search implementation
  - `merge.py` - Result merging utilities (RRF, score-based)
  - `embeddings.py` - HuggingFace embeddings
  - `retrieval.py` - MMR and reranking functions
  - `pipeline.py` - Indexer and RAG classes
  - `llm.py` - LLM interface (Gemini/HuggingFace)
  - `loaders.py` - Document loaders
  - `tools.py` - Web search tools
  - `config.py` - Configuration management
- `scripts/` runnable scripts to showcase steps
  - `download_sample_data.py` - Download from Internet Archive
  - `extraction_wikipedia.py` - Extract Wikipedia pages
  - `index_docs.py` - Index documents with chosen chunking strategy
  - `inspect_chroma.py` - Inspect database contents and statistics
  - `query_basic.py` - Basic RAG query
  - `query_enhanced.py` - Enhanced RAG with HyDE, MMR, reranking
  - `query_agentic.py` - Agentic RAG with decision routing
  - `query_hybrid.py` - Query with different retrieval modes
  - `compare_retrieval.py` - Compare vector/BM25/hybrid retrieval
  - `eval_ragas.py` - Evaluate RAG quality with RAGAS metrics
  - `compare_chunking.py` - Compare chunking strategies
- `data/docs/` documents to index
- `.chroma/` local vector DB persistence (created after indexing)
  - `chroma.sqlite3` - Vector store database
  - `bm25_docs.pkl` - BM25 index (keyword search)

## Command Reference

### Data Preparation

#### Extract Wikipedia Pages

```bash
python scripts/extraction_wikipedia.py
# Saves pages to data/docs/wikipedia as text files
```

#### Import FineWiki dataset

```bash
python scripts/download_wiki_dataset.py
# Saves pages to data/docs/wiki_finewiki_en as Markdown files
```

### Indexing Options

```bash
python scripts/index_docs.py [OPTIONS]

Options:
  --folder PATH           Path to documents folder (default: data/docs)
  --chunking STRATEGY     Chunking strategy: fixed, recursive, semantic (default: fixed)
  --persist PATH          Chroma persistence directory (default: .chroma)
```

**Examples:**
```bash
# Index with fixed chunking
python scripts/index_docs.py --chunking fixed

# Index with semantic chunking (adds context headers and LLM summaries)
python scripts/index_docs.py --chunking semantic

# Index custom folder
python scripts/index_docs.py --folder my_docs --chunking recursive
```

### Query Options

#### Basic Query

```bash
python scripts/query_basic.py [OPTIONS]

Options:
  --question TEXT         Question to ask (required)
  --top_k INT            Number of results (default: 4)
  --persist PATH         Chroma directory (default: .chroma)
  --dry-run              Show chunks without LLM call
```

#### Hybrid Retrieval Query

```bash
python scripts/query_hybrid.py [OPTIONS]

Options:
  --question TEXT         Question to ask (required)
  --mode MODE            Retrieval mode: vector, bm25, hybrid (default: hybrid)
  --top_k INT            Number of results (default: 4)
  --persist PATH         Chroma directory (default: .chroma)
  --dry-run              Show chunks without LLM call
```

#### Enhanced Query

```bash
python scripts/query_enhanced.py [OPTIONS]

Options:
  --question TEXT         Question to ask (required)
  --persist PATH         Chroma directory (default: .chroma)
  --hyde                 Use Hypothetical Document Embeddings
  --mmr                  Use MMR for diversity
  --rerank               Use LLM reranking
  --filter METADATA      Metadata filter (format: key=value;key2=value2)
```

#### Agentic Query

```bash
python scripts/query_agentic.py [OPTIONS]

Options:
  --question TEXT         Question to ask (required)
  --persist PATH         Chroma directory (default: .chroma)
```

#### Compare Retrieval Modes (CLI)

```bash
python scripts/compare_retrieval.py [OPTIONS]

Options:
  --question TEXT         Question to ask (required)
  --top_k INT            Results per mode (default: 3)
  --persist PATH         Chroma directory (default: .chroma)
```

#### Inspect Database

```bash
python scripts/inspect_chroma.py [OPTIONS]

Options:
  --persist PATH         Chroma directory (default: .chroma)
  --collection NAME      Collection name (default: docs)
  --sample INT           Number of sample docs to show (default: 3)
```


## Tips and Best Practices

### Chunking Strategy Selection

- Use **fixed** for general documents, fastest processing
- Use **recursive** for better semantic preservation in paragraphs
- Use **semantic** for structured documents with clear sections (Wikipedia, technical docs)

### Retrieval Mode Selection

- Use **hybrid** (default) for best results - combines semantic + keyword search
- Use **vector** for conceptual/semantic queries
- Use **bm25** for exact keyword/term matching

### Performance Optimization

- Enable MMR (`--mmr`) to increase diversity and reduce redundancy
- Use reranking (`--rerank`) for highest quality but slower results
- Use metadata filters to narrow search scope
- Adjust `top_k` based on context window and quality needs

### Troubleshooting

- Use `--dry-run` to inspect retrieved chunks without LLM costs
- Use `inspect_chroma.py` to verify indexing worked correctly
- Check `.env` file for correct API keys
- Ensure BM25 index exists (created automatically during indexing)

### Web search notes

- DuckDuckGo can rate limit frequent queries. If you momentarily get 0 results, wait a few seconds and retry.
- You may see a warning: "This package (duckduckgo_search) has been renamed to ddgs". The current dependency works; optionally install `ddgs`.
- The code automatically simplifies very long or question-like queries to improve result quality.

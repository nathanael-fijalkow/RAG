from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich import print
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import RAG


def load_questions(path: Path) -> Tuple[List[str], Optional[List[str]]]:
    """
    Load questions (and optional ground truth references) from a file.
    Supports:
      - .txt (one question per line)
      - .csv (columns: question, ground_truth [optional])
    Returns (questions, references) where references can be None.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".txt":
        questions = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return questions, None

    if path.suffix.lower() == ".csv":
        questions: List[str] = []
        refs: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "question" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'question' column")
            for row in reader:
                q = (row.get("question") or "").strip()
                if not q:
                    continue
                questions.append(q)
                gt = (row.get("ground_truth") or "").strip()
                refs.append(gt)
        # If no non-empty refs, return None
        if any(r for r in refs):
            return questions, refs
        return questions, None

    raise ValueError("Unsupported file extension. Use .txt or .csv")


def build_dataset_records(
    rag: RAG,
    questions: List[str],
    references: Optional[List[str]] = None,
    top_k: int = 4,
    retrieval_mode: str = "hybrid",
    use_mmr: bool = True,
    use_rerank: bool = False,
) -> Dict[str, List]:
    """
    Run retrieval + answer generation, and build records compatible with RAGAS evaluate().
    RAGAS expects a dataset with columns: question, answer, contexts (list[str]), ground_truth (list[str], optional)
    """
    all_questions: List[str] = []
    all_answers: List[str] = []
    all_contexts: List[List[str]] = []
    all_gts: List[List[str]] = []

    for i, q in enumerate(questions, 1):
        print(f"[RAGAS] Processing {i}/{len(questions)}: {q}")
        docs, metas = rag.retrieve(
            q,
            top_k=top_k,
            use_mmr=use_mmr,
            use_rerank=use_rerank,
            retrieval_mode=retrieval_mode,
        )
        if not docs:
            # ensure shapes match
            all_questions.append(q)
            all_answers.append("")
            all_contexts.append([])
            if references is not None:
                gt = references[i - 1] if i - 1 < len(references) else ""
                all_gts.append([gt] if gt else [])
            continue

        answer = rag.answer(q, docs, metas)
        all_questions.append(q)
        all_answers.append(answer)
        all_contexts.append(docs)
        if references is not None:
            gt = references[i - 1] if i - 1 < len(references) else ""
            all_gts.append([gt] if gt else [])

    data = {
        "question": all_questions,
        "answer": all_answers,
        "contexts": all_contexts,
    }
    if references is not None:
        data["ground_truth"] = all_gts
    return data


def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
    ap.add_argument("--questions", type=str, required=True, help="Path to questions file (.txt or .csv). CSV may include ground_truth column.")
    ap.add_argument("--persist", type=str, default=str(Path(__file__).resolve().parents[1] / ".chroma"), help="Chroma persist directory")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid", help="Retrieval mode")
    ap.add_argument("--top_k", type=int, default=4, help="Number of contexts to retrieve")
    ap.add_argument("--no-mmr", action="store_true", help="Disable MMR diversity")
    ap.add_argument("--rerank", action="store_true", help="Enable LLM reranking before evaluation")
    ap.add_argument("--out", type=str, default="eval_ragas_report.json", help="Output JSON report path")
    args = ap.parse_args()

    questions_path = Path(args.questions)
    # If only a filename (no directory), look in data/eval/ by default
    if not questions_path.is_absolute() and questions_path.parent == Path("."):
        questions_path = Path(__file__).resolve().parents[1] / "data" / "eval" / questions_path.name
    questions, references = load_questions(questions_path)

    rag = RAG(persist=args.persist, use_bm25=True)

    # Build dataset
    records = build_dataset_records(
        rag,
        questions,
        references=references,
        top_k=args.top_k,
        retrieval_mode=args.mode,
        use_mmr=not args.no_mmr,
        use_rerank=args.rerank,
    )

    # Convert to HF Dataset
    from datasets import Dataset
    
    # RAGAS 0.3+ expects specific column names
    # Rename 'contexts' to 'retrieved_contexts' if needed
    if 'contexts' in records and 'retrieved_contexts' not in records:
        records['retrieved_contexts'] = records.pop('contexts')
    
    # RAGAS 0.3+ expects "reference" as a STRING (not list) and "ground_truth" must be removed
    if 'ground_truth' in records:
        # Convert ground_truth lists to strings for RAGAS "reference" field
        ground_truth_strings = []
        for gt in records['ground_truth']:
            if isinstance(gt, list):
                # If it's a list, take the first element or empty string
                ground_truth_strings.append(gt[0] if gt else "")
            else:
                ground_truth_strings.append(str(gt) if gt else "")
        records['reference'] = ground_truth_strings
        del records['ground_truth']  # Remove the old key
    
    ds = Dataset.from_dict(records)

    # Select metrics for RAGAS 0.3+
    from ragas.metrics import (
        context_precision,
        context_recall,
        answer_relevancy,
        faithfulness,
    )
    metrics = [context_precision, answer_relevancy, faithfulness]
    if "ground_truth" in ds.column_names:
        metrics.append(context_recall)  # recall needs ground_truth
        try:
            from ragas.metrics import answer_correctness  # may not be available in all versions
            metrics.append(answer_correctness)
        except Exception:
            pass

    # Configure RAGAS to use Gemini instead of OpenAI (for RAGAS 0.3+)
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    
    # Get settings for API key
    from src.config import get_settings
    settings = get_settings()
    
    # Create Gemini LLM for RAGAS
    langchain_llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0
    )
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    
    # Create Gemini Embeddings for RAGAS
    langchain_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.gemini_api_key
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    # Evaluate with RAGAS
    from ragas import evaluate
    result = evaluate(ds, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)

    # Print and save - RAGAS 0.3+ returns EvaluationResult object
    print("\n[bold]RAGAS Evaluation Results[/bold]")
    
    # Access results from the EvaluationResult object
    # In RAGAS 0.3+, result has a 'scores' attribute which is a list of dicts
    # and a 'to_pandas()' method which returns a DataFrame
    try:
        # Get DataFrame with all scores
        df = result.to_pandas()
        
        # Select only numeric columns (the actual metrics)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            # Calculate mean scores across all samples
            scores = df[numeric_cols].mean().to_dict()
            
            print("\n" + "="*50)
            for k, v in scores.items():
                print(f"- {k}: {v:.4f}")
            print("="*50)

            out_path = Path(args.out)
            out_payload = {k: float(v) for k, v in scores.items()}
            out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
            print(f"\nSaved report to: {out_path}")
        else:
            print("[WARNING] No numeric metric columns found in RAGAS result")
            
    except Exception as e:
        print(f"[ERROR] Failed to process RAGAS results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

import os
import sys
import types
import pyarrow.parquet as pq


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.dataset_create import all_qa_pairs, print_qa_pairs
from evaluation.dataset_create import OLLAMA_MODEL

DEFAULT_EVAL_LIMIT = os.getenv("RAGAS_EVAL_LIMIT")
DEFAULT_EVAL_LIMIT = int(DEFAULT_EVAL_LIMIT) if DEFAULT_EVAL_LIMIT else None
DEFAULT_RAGAS_TIMEOUT_SECONDS = int(os.getenv("RAGAS_TIMEOUT_SECONDS", "600"))
DRY_RUN = os.getenv("RAGAS_DRY_RUN", "0") == "1"


def patch_ragas_vertexai_import():
    """RAGAS imports this optional module at import time in some versions."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    vertexai_module = types.ModuleType(module_name)

    class ChatVertexAI:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChatVertexAI is not installed and is not used by this project.")

    vertexai_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = vertexai_module


def summarize_scores(df):
    scores = {}
    skipped_columns = {
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
    }

    for column in df.columns:
        if column in skipped_columns:
            continue

        try:
            numeric_values = df[column].dropna().astype(float)
        except (TypeError, ValueError):
            continue

        if not numeric_values.empty:
            scores[column] = numeric_values.mean()

    return scores


def run_ragas_evaluation(qa_pairs=None, output_csv_path="ragas_results.csv", limit=None):
    qa_pairs = qa_pairs if qa_pairs is not None else all_qa_pairs
    if limit is not None:
        qa_pairs = qa_pairs[:limit]

    if not qa_pairs:
        raise RuntimeError(
            "No QA dataset found. Run this first: python -m evaluation.dataset_create"
        )

    print(f"Preparing RAGAS dataset with {len(qa_pairs)} rows...", flush=True)

    from datasets import Dataset

    dataset = Dataset.from_dict({
        "user_input": [
            row.get("question", "")
            for row in qa_pairs
        ],

        "response": [
            row.get("answer", row.get("ground_truth", ""))
            for row in qa_pairs
        ],

        "retrieved_contexts": [
            row.get("contexts", [])
            for row in qa_pairs
        ],

        "reference": [
            row.get("ground_truth", "")
            for row in qa_pairs
        ]
    })

    print(f"Using Ollama model for RAGAS: {OLLAMA_MODEL}", flush=True)

    from langchain_ollama import ChatOllama, OllamaEmbeddings

    evaluator_llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        timeout=DEFAULT_RAGAS_TIMEOUT_SECONDS,
    )
    evaluator_embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    patch_ragas_vertexai_import()

    print("Importing RAGAS...", flush=True)
    from ragas import evaluate
    from ragas.run_config import RunConfig

    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall
    )

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ]

    print("Running RAGAS evaluation. This can take a while with local Ollama...", flush=True)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(
            timeout=DEFAULT_RAGAS_TIMEOUT_SECONDS,
            max_retries=1,
            max_workers=1,
        ),
        batch_size=1,
        raise_exceptions=False,
    )

    print("Saving RAGAS results...", flush=True)
    df = result.to_pandas()
    df.to_csv(output_csv_path, index=False)

    return result, df, summarize_scores(df)


def main():
    print("Starting RAGAS evaluation script...", flush=True)

    if not all_qa_pairs:
        raise RuntimeError(
            "No QA dataset found. Run this first: python -m evaluation.dataset_create"
        )

    print(f"Loaded {len(all_qa_pairs)} QA rows from generated_qa.json", flush=True)

    if DRY_RUN:
        print("Dry run enabled. Dataset loaded successfully; evaluation was skipped.", flush=True)
        return

    result, df, scores = run_ragas_evaluation(all_qa_pairs, limit=DEFAULT_EVAL_LIMIT)

    print(result)
    print(df.head())

    if scores:
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("No numeric RAGAS scores were produced. Check warnings/timeouts above.")



if __name__ == "__main__":
    main()

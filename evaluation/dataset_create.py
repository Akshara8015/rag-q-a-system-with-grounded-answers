import json
import os
import re
import sys
import time
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
DEFAULT_DELAY_SECONDS = 0
DEFAULT_RETRIES = 3
DEFAULT_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "pdf1.pdf")
DATASET_CACHE_PATH = os.path.join(PROJECT_ROOT, "evaluation", "generated_qa.json")


def generate_with_ollama(system_prompt: str, model: str = OLLAMA_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": system_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()

def prompt(chunk):
    system_prompt = f"""
            You are an expert dataset generation assistant specializing in creating evaluation datasets for Retrieval-Augmented Generation (RAG) systems.
            Your task is to analyze the provided document and generate a comprehensive set of question-answer pairs that can be used to evaluate a RAG pipeline.

            Requirements:
            1. Generate diverse questions covering the entire document.
            2. Questions must be answerable solely from the document.
            3. Do not use external knowledge.
            4. Answers must be factually correct and directly supported by the document.
            5. Generate questions of varying difficulty:

            * Easy factual questions
            * Medium conceptual questions
            * Numerical questions
            * Definition questions
            * Comparison questions
            * Reasoning questions
            * Section-specific questions
            * Multi-hop questions requiring information from multiple parts of the document

            6. Avoid duplicate or highly similar questions.
            7. Ensure coverage of all major topics, sections, tables, figures, formulas, results, and conclusions present in the document.
            8. Answers should be concise but complete.
            9. If a page number is available, include it.
            10. Generate evaluation-quality data rather than trivia.

            Output Format:

            [
                {{
                "question": "...",
                "ground_truth": "...",
                "source_page": "..."
                }}
            ]

            Example:

            [
                {{
                "question": "What is self-attention?",
                "ground_truth": "Self-attention is an attention mechanism that relates different positions of a single sequence to compute a representation of the sequence.",
                "source_page": "3"
                }},
                {{
                "question": "Why is the dot product scaled in scaled dot-product attention?",
                "ground_truth": "The dot product is scaled by the square root of the key dimension to prevent extremely small gradients caused by large dot-product values.",
                "source_page": "4"
                }}
            ]



            Generate as 2 high-quality question-answer pairs as possible from the provided document without sacrificing quality.

            text:
            {chunk['text']}

            Return ONLY valid JSON. Do not include markdown, commentary, or explanations.

            """
    
    return system_prompt

            # Question Distribution Guidelines:

            # * 20% factual questions
            # * 20% conceptual questions
            # * 15% numerical questions
            # * 15% reasoning questions
            # * 10% comparison questions
            # * 10% definition questions
            # * 10% multi-hop questions


def generate_content_with_retry(system_prompt: str, retries: int = DEFAULT_RETRIES):
    for attempt in range(retries):
        try:
            return generate_with_ollama(system_prompt)
        except Exception as exc:
            if attempt == retries - 1:
                raise

            wait_seconds = 2 ** attempt
            print(f"Ollama generation failed. Retrying in {wait_seconds} seconds... Error: {exc}")
            time.sleep(wait_seconds)


def parse_qa_pairs(content: str):
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        json_array_match = re.search(r"\[[\s\S]*\]", content)
        if not json_array_match:
            raise
        parsed = json.loads(json_array_match.group(0))

    if isinstance(parsed, dict):
        if "question" in parsed and "ground_truth" in parsed:
            return [parsed]

        for key in ("qa_pairs", "questions", "data", "items"):
            if isinstance(parsed.get(key), list):
                return parsed[key]

    if isinstance(parsed, list):
        return parsed

    raise ValueError("Ollama response did not contain a JSON list of QA pairs.")


def create_dataset(
    file_path: str,
    max_chunks: int | None = None,
    delay_seconds: int = DEFAULT_DELAY_SECONDS,
):
    from rag.ocr import extract_text_from_pdf

    chunks = extract_text_from_pdf(file_path)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    print("got chunks")

    all_qa_pairs = []

    for chunk in chunks:
        system_prompt = prompt(chunk)
        content = generate_content_with_retry(system_prompt)
        # print(content)

        try:
            qa_pairs = parse_qa_pairs(content)
            for qa in qa_pairs:
                qa["source_page"] = chunk["metadata"].get("page", "")
                qa["contexts"] = [chunk["text"]]
                qa["answer"] = qa.get("ground_truth", "")
            all_qa_pairs.extend(qa_pairs)

        except (json.JSONDecodeError, ValueError):
            print(
                f"Failed to parse JSON for page "
                f"{chunk['metadata'].get('page')}"
            )
            # print(content)
            continue

        if delay_seconds:
            time.sleep(delay_seconds)

    return all_qa_pairs


def load_saved_dataset(cache_path: str = DATASET_CACHE_PATH):
    if not os.path.exists(cache_path):
        return []

    with open(cache_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_dataset(qa_pairs, cache_path: str = DATASET_CACHE_PATH):
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(qa_pairs, file, indent=2)


def create_and_save_dataset(file_path: str = DEFAULT_FILE_PATH):
    global all_qa_pairs

    all_qa_pairs = create_dataset(file_path)
    save_dataset(all_qa_pairs)
    return all_qa_pairs


def print_qa_pairs(all_qa_pairs):
    for qa_pair in all_qa_pairs:
        print("Question:", qa_pair.get("question", ""))
        print("Answer:", qa_pair.get("ground_truth", ""))
        print("Source Page:", qa_pair.get("source_page", ""))
        print("-" * 80)


all_qa_pairs = load_saved_dataset()


if __name__ == "__main__":
    all_qa_pairs = create_and_save_dataset(DEFAULT_FILE_PATH)
    # print_qa_pairs(all_qa_pairs)


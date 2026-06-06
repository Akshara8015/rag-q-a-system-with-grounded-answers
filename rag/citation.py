import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .ocr import *
from .HybridRetriever import *
from .utils import *


def get_config_value(name: str, default: str = ""):
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default

# ==============================
# Load Vector Store + Retriever
# (left as-is — can be wired into HybridRetriever)
# ==============================

def build_context(retrieved_docs):
    return "\n\n".join([
        f"[Page {doc['metadata'].get('page', 'unknown')}]\n{doc['text']}"
        for doc in retrieved_docs
    ])


def load_retriever(file_path: str, chroma_dir: str = "./chroma_db"):
    chunks = extract_text_from_pdf(file_path)
    retriever = HybridRetriever(chunks, chroma_dir=chroma_dir)
    return retriever, chunks


# ==============================
#           LLM 
# ==============================
def build_llm():
    openai_api_key = get_config_value("OPENAI_API_KEY")
    if not openai_api_key:
        return None

    return ChatOpenAI(
        model=get_config_value("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=openai_api_key,
        temperature=0.7,
    )


llm = build_llm()

system_prompt = """
                You are an expert document question-answering assistant.

                Your task is to answer questions ONLY using the provided context.

                RULES:

                1. Use ONLY information present in the context.
                2. Do NOT use prior knowledge.
                3. If the answer is not found in the context, reply:
                "I could not find this information in the uploaded document."

                4. Every factual statement MUST include a citation.

                5. Citations must use the format:
                [Page X]

                Examples:
                - The refund period is 30 days. [Page 5]
                - BM25 is a lexical retrieval algorithm. [Page 12]

                6. If information comes from multiple pages:
                [Page 4, Page 7]

                7. Do NOT invent citations.

                8. Keep answers concise but complete.

                9. Never mention:
                - "According to the provided context"
                - "The context states"
                - "Based on the retrieved chunks"

                10. Present the answer naturally as if speaking directly to the user.

            """


def answer_query(query: str, context: str = "") -> str:
    """Build messages and call the LLM to produce a grounded answer.

    Parameters:
    - query: user question
    - context: retrieved context (plain text) to ground the answer

    Returns:
    - LLM answer content (string)
    """

    if not query or not isinstance(query, str):
        return "I could not find this information in the uploaded document."

    user_prompt = f"""
                    Question:
                    {query}
                    Retrieved Context:
                    {context}
                    Answer:
                """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    if llm is None:
        return "I could not find this information in the uploaded document."

    try:
        response = llm.invoke(messages)
        return getattr(response, "content", str(response))
    except Exception:
        return "I could not find this information in the uploaded document."


if __name__ == "__main__":
    sample_pdf = os.path.join(os.path.dirname(__file__), "..", "data", "pdf1.pdf")
    sample_pdf = os.path.abspath(sample_pdf)
    retriever, _ = load_retriever(sample_pdf)
    retrieved_docs = retriever.retrieve("What is BM25?")
    context = build_context(retrieved_docs)
    print(answer_query("What is BM25?", context))
#     # simple local test
#     sample_context = "[Page 2]\nBM25 is a ranking function used by search engines to estimate relevance."
#     print(answer_query("What is BM25?", sample_context))


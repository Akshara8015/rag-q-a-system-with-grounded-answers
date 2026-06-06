# Streamlit Deployment

This app is ready for Streamlit deployment with Chroma Cloud.

## Required Secrets

Set these in Streamlit Community Cloud under app settings:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = qwen/qwen3-32b
CHROMA_API_KEY = "your_chroma_api_key_here"
CHROMA_TENANT = "your_chroma_tenant_here"
CHROMA_DATABASE = "your_chroma_database_here"
CHROMA_COLLECTION_PREFIX = "document_chunks"
```

Use your real TryChroma values in the hosted secrets UI, not in committed files.

## Entry Point

```text
app.py
```

## How Chroma Works

The app uses `rag/utils.py` to create a Python `chromadb.CloudClient` when all three Chroma values are present. Uploaded PDFs are split into chunks and stored in a per-upload Chroma collection named like:

```text
document_chunks_<upload_hash>
```

If Chroma Cloud variables are missing in a hosted environment, the app raises a clear configuration error instead of writing to local disk.

## Local Development

```powershell
python -m venv rag_env2
.\rag_env2\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

## Render Alternative

`render.yaml` is still included if you deploy through Render instead of Streamlit Community Cloud. Add the same environment variables in Render.

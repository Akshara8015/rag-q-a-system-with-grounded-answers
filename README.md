# Enterprise Document Intelligence

Streamlit app for uploading PDFs and asking grounded questions using hybrid retrieval, Chroma Cloud, and OpenAI.

## Run Locally

```powershell
python -m venv rag_env2
.\rag_env2\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Add these values to `.env` for local development:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_chroma_tenant_here
CHROMA_DATABASE=your_chroma_database_here
```

## Deploy With Streamlit

1. Push this repository to GitHub.
2. Create a Streamlit app with `app.py` as the entrypoint.
3. Add the values from `.streamlit/secrets.toml.example` to Streamlit secrets.
4. Deploy.

The app creates a Python Chroma Cloud client in `rag/utils.py`. A JavaScript `config/chroma.js` file is not needed for this Streamlit project.

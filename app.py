import streamlit as st
import os
import hashlib
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from rag.ocr import extract_text_from_pdf
from rag.HybridRetriever import HybridRetriever


load_dotenv()


def get_config_value(name, default=""):
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def collection_name_for_upload(uploaded_file_id):
    digest = hashlib.sha1(uploaded_file_id.encode("utf-8")).hexdigest()[:16]
    prefix = get_config_value("CHROMA_COLLECTION_PREFIX", "document_chunks")
    return f"{prefix}_{digest}"


def build_llm():
    groq_api_key = get_config_value("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. Add it to Streamlit secrets or "
            "your deployment environment variables."
        )

    return ChatOpenAI(
        model=get_config_value("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=groq_api_key,
        temperature=0.7,
    )

# Page configuration
st.set_page_config(
    page_title="Document Intelligence Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    :root {
        --primary-color: #6366f1;
        --secondary-color: #ec4899;
        --accent-color: #f59e0b;
        --success-color: #10b981;
        --danger-color: #ef4444;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f8f9ff 0%, #fef3f4 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    /* Custom title */
    .title-container {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
    }
    
    .title-container h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .title-container p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #ec4899;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #ecf0ff 0%, #f3e8ff 100%);
        padding: 1rem;
        border-radius: 12px;
        color: #1f2937;
        margin: 0.5rem 0;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #6366f1;
        border-radius: 12px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #ddd6fe 100%);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Spinner animation */
    .stSpinner {
        color: #6366f1;
    }
    
    /* Text input styling */
    .stTextInput>div>div>input {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe8f0 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

if "chroma_collection_name" not in st.session_state:
    st.session_state.chroma_collection_name = None

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("---")
    st.markdown("### 📁 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF document to ask questions about it"
    )
    
    if uploaded_file is not None and (
        st.session_state.uploaded_file_id != f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
        or st.session_state.retriever is None
        or st.session_state.chunks is None
    ):
        uploaded_file_id = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
        is_new_upload = st.session_state.uploaded_file_id != uploaded_file_id

        # Save uploaded file temporarily (cross-platform)
        import tempfile

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("📚 Processing PDF and building retriever..."):
            try:
                # Extract chunks from PDF
                chunks = extract_text_from_pdf(temp_path)
                collection_name = collection_name_for_upload(uploaded_file_id)
                
                # Initialize retriever
                retriever = HybridRetriever(chunks, collection_name=collection_name)
                
                # Store in session state
                st.session_state.retriever = retriever
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chunks = chunks
                st.session_state.chroma_collection_name = collection_name
                if is_new_upload:
                    st.session_state.chat_history = []  # Reset chat only when the PDF changes
                st.session_state.uploaded_file_id = uploaded_file_id
                
                st.success(f"✅ Successfully loaded '{uploaded_file.name}'")
                st.info(f"📊 Extracted {len(chunks)} chunks from the PDF")
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📋 Session Info")
    
    if st.session_state.pdf_name:
        st.write(f"**Current PDF:** {st.session_state.pdf_name}")
        st.write(f"**Chunks:** {len(st.session_state.chunks) if st.session_state.chunks else 0}")
    else:
        st.write("📝 No PDF loaded yet")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.markdown("""
<div class="title-container">
    <h1>📄 Document Intelligence Chat</h1>
    <p>Ask questions about your PDF documents with AI-powered retrieval</p>
</div>
""", unsafe_allow_html=True)

# Check if PDF is loaded
if st.session_state.retriever is None:
    st.warning("⚠️ Please upload a PDF file to get started!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🚀 How It Works
        1. Upload a PDF
        2. Ask any question
        3. Get AI answers with citations
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Features
        - Hybrid search (vector + BM25)
        - AI-powered retrieval
        - Page citations
        - Multi-turn chat
        """)
    
    with col3:
        st.markdown("""
        ### 🔧 Technology
        - RAG Pipeline
        - Embedding Models
        - OpenAI chat model
        - Chroma DB
        """)
else:
    # Display chat history
    st.markdown("### 💬 Chat History")
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <b>You:</b> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <b>Assistant:</b> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    st.markdown("### ❓ Ask a Question")
    
    col1, col2 = st.columns([0.9, 0.1])
    
    with col1:
        user_query = st.text_input(
            "Type your question here...",
            placeholder="E.g., What is the main topic of this document?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    
    if send_button and user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        with st.spinner("🔍 Retrieving information and generating answer..."):
            try:
                # Rebuild the retriever if it was not persisted across reruns
                if st.session_state.retriever is None and st.session_state.chunks:
                    st.session_state.retriever = HybridRetriever(
                        st.session_state.chunks,
                        collection_name=st.session_state.chroma_collection_name or "document_chunks",
                    )

                if st.session_state.retriever is None:
                    raise ValueError("No retriever available. Please upload a PDF first.")

                # Initialize LLM from deployment secrets/environment.
                llm = build_llm()
                
                # Retrieve final chunks through HybridRetriever.retrieve()
                final_chunks = st.session_state.retriever.retrieve(user_query, final_k=5)
                if not final_chunks:
                    assistant_response = "I could not find this information in the uploaded document."
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    st.warning("⚠️ No relevant passages were found for that question. Try a different query.")
                    st.rerun()

                # Build context
                context = "\n\n".join([
                    f"[Page {doc['metadata'].get('page', 'unknown')}]\n{doc['text']}"
                    for doc in final_chunks
                ])
                
                # System prompt
                system_prompt = """You are an expert document question-answering assistant.

                                Your task is to answer questions ONLY using the provided context.

                                RULES:
                                1. Use ONLY information present in the context.
                                2. Do NOT use prior knowledge.
                                3. If the answer is not found in the context, reply: "I could not find this information in the uploaded document."
                                4. Every factual statement MUST include a citation.
                                5. Citations must use the format: [Page X]
                                6. If information comes from multiple pages: [Page 4, Page 7]
                                7. Do NOT invent citations.
                                8. Keep answers concise but complete.
                                9. Never mention "According to the provided context" or similar phrases.
                                10. Present the answer naturally as if speaking directly to the user."""
                                            
                # Prepare messages
                user_prompt = f"""Question: {user_query}
                                Retrieved Context:
                                {context}
                                Answer:"""
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                # Get response
                try:
                    response = llm.invoke(messages)
                    assistant_response = getattr(response, "content", str(response)).strip()
                except Exception as llm_error:
                    assistant_response = (
                        "I retrieved relevant passages, but the LLM call failed, so I could not generate "
                        f"a final answer. Error: {llm_error}"
                    )

                if not assistant_response:
                    assistant_response = "I could not find this information in the uploaded document."
                
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                # Remove the user message if there was an error
                st.session_state.chat_history.pop()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🚀 Powered by Hybrid RAG | OpenAI | Chroma DB</p>
    <p style="font-size: 0.9rem; opacity: 0.7;">Built for Enterprise Document Intelligence</p>
</div>
""", unsafe_allow_html=True)

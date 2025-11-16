"""
VectorFlow-RAG Interactive Demo
Streamlit web interface for semantic search and RAG
"""
import streamlit as st
import sys, os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline
import time

# Check if running locally or on Streamlit Cloud
if "STREAMLIT_CLOUD" in os.environ:
    st.warning("⚠️ NOTE: Streamlit Cloud deployment requires Ollama running on your local machine or a remote server. For full functionality, run this app locally with: `ollama serve`")
else:
    # Local mode - check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            st.sidebar.error("❌ Ollama not running! Start with: ollama serve")
    except:
        st.sidebar.error("❌ Ollama not reachable at http://localhost:11434")

# Page configuration
st.set_page_config(
    page_title="VectorFlow-RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --surface: #ffffff;
        --surface-raised: #f8fafc;
        --border: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --accent: #0ea5e9;
        --success: #10b981;
        --warning: #f59e0b;
    }
    
    * {
        font-family: 'IBM Plex Sans', -apple-system, system-ui, sans-serif;
    }
    
    .stApp {
        background: #fafbfc;
    }
    
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 3rem 4rem 3rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .answer-box {
        background: linear-gradient(to bottom, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #93c5fd;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #1e3a8a;
    }
    
    .source-doc {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--primary);
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        font-size: 0.95rem;
        line-height: 1.65;
        color: var(--text-primary);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.85rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid;
    }
    
    .badge-success {
        background: #d1fae5;
        color: #065f46;
        border-color: #6ee7b7;
    }
    
    .badge-warning {
        background: #fed7aa;
        color: #92400e;
        border-color: #fdba74;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: transparent;
        border-bottom: 2px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-radius: 0;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 0 1.5rem;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary);
        border-bottom-color: var(--primary);
    }
    
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.75rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: var(--primary-dark);
        transform: translateY(-1px);
    }
    
    .stTextInput input, .stTextArea textarea {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.7rem 1rem;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="metric-container"] {
        background: var(--surface-raised);
        border: 1px solid var(--border);
        padding: 1rem;
        border-radius: 8px;
    }
    
    .streamlit-expanderHeader {
        background: var(--surface-raised);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stProgress > div > div > div > div {
        background: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-header">🔍 VectorFlow-RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade semantic search and RAG system</div>', unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model Settings")
    
    llm_model = st.selectbox(
        "LLM Model",
        ["tinyllama", "llama3.2:1b", "mistral:7b"],
        help="Ollama model for generation"
    )
    
    alpha = st.slider(
        "Hybrid Search Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = BM25 only, 1 = Vector only"
    )
    
    k_docs = st.slider(
        "Context Documents",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of documents for RAG"
    )
    
    st.markdown("---")
    
    st.subheader("📊 System Status")
    
    if st.session_state.documents_loaded:
        st.markdown('<div class="status-badge badge-success">✓ System Ready</div>', unsafe_allow_html=True)
        if st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_stats()
            st.metric("Documents", stats['documents_ingested'])
            st.metric("Chunks", stats['chunks_indexed'])
            st.metric("Embedding Dim", stats['embedding_dimension'])
    else:
        st.markdown('<div class="status-badge badge-warning">⚠ No Data Loaded</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.markdown("""
    **VectorFlow-RAG** is a modular semantic search and RAG framework.
    
    **Features:**
    - Hybrid search (BM25 + vectors)
    - Local LLM inference
    - Modular architecture
    - 100% free and open source
    
    **Stack:** Sentence Transformers, ChromaDB, BM25S, Ollama, Streamlit
    """)

# =============================================================================
# MAIN CONTENT
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📝 Document Upload", "🔍 Search", "💬 Ask Questions"])

# -----------------------------------------------------------------------------
# TAB 1: Document Upload
# -----------------------------------------------------------------------------
with tab1:
    st.header("Upload & Index Documents")
    st.write("Add documents to your knowledge base for semantic search and Q&A.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        use_sample = st.checkbox("Use sample documents", value=True)
        
        if use_sample:
            st.info("📚 Using built-in sample documents about VectorFlow-RAG")
            sample_docs = [
                """VectorFlow-RAG is a production-grade, modular semantic retrieval and RAG framework.
                It is designed for machine learning engineers and researchers who want full transparency 
                and control over their search pipeline. The system is built to be reproducible, swappable, 
                and benchmarkable with support for various embedding models and retrieval strategies.""",
                
                """The embedding layer uses state-of-the-art models from sentence-transformers.
                By default, it uses all-MiniLM-L6-v2, a lightweight 80MB model with 384 dimensions.
                Users can easily swap embedding models through configuration. The embeddings capture 
                semantic meaning and enable similarity-based search.""",
                
                """Hybrid search combines BM25 lexical matching with vector similarity search. 
                BM25 handles exact keyword matches well, while vector search captures semantic 
                similarity. The alpha parameter controls the balance between the two approaches.
                Alpha=0.5 gives equal weight to both methods for optimal results.""",
                
                """VectorFlow-RAG uses Ollama for local LLM inference, ensuring privacy and zero API costs.
                Supported models include TinyLlama (600MB), Llama 3.2 (1.3GB), and Mistral (4GB).
                All models run on consumer hardware. The system combines retrieved context with LLM 
                generation to produce grounded, factual answers.""",
                
                """The framework includes comprehensive experiment tracking using MLflow and DagsHub.
                All experiments log metrics like NDCG, MRR, recall@k, and latency automatically.
                This makes VectorFlow-RAG suitable for research and systematic ablation studies.
                Users can compare different configurations and reproduce results.""",
            ]
        else:
            num_docs = st.number_input("Number of documents", min_value=1, max_value=10, value=3)
            sample_docs = []
            for i in range(num_docs):
                doc = st.text_area(f"Document {i+1}", height=150, key=f"doc_{i}")
                if doc:
                    sample_docs.append(doc)
    
    with col2:
        st.subheader("Quick Stats")
        if sample_docs:
            total_chars = sum(len(doc) for doc in sample_docs)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(sample_docs)}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_chars:,}</div>
                <div class="stat-label">Characters</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_chars // len(sample_docs):,}</div>
                <div class="stat-label">Avg Length</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Index Documents", type="primary", use_container_width=True):
        if not sample_docs or all(not doc.strip() for doc in sample_docs):
            st.error("Please provide at least one document")
        else:
            with st.spinner("Processing documents..."):
                try:
                    rag = RAGPipeline(
                        index_dir="indices\\streamlit_rag",
                        alpha=alpha,
                        llm_model=llm_model
                    )
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Chunking documents...")
                    progress_bar.progress(25)
                    time.sleep(0.5)
                    
                    status_text.text("Generating embeddings...")
                    progress_bar.progress(50)
                    
                    rag.ingest_documents(sample_docs, reset=True)
                    
                    progress_bar.progress(75)
                    status_text.text("Building indices...")
                    time.sleep(0.5)
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    st.session_state.rag_pipeline = rag
                    st.session_state.documents_loaded = True
                    
                    st.success("✓ Documents indexed successfully")
                    st.balloons()
                    
                    stats = rag.get_stats()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Documents", stats['documents_ingested'])
                    col2.metric("Chunks", stats['chunks_indexed'])
                    col3.metric("Embedding Dim", stats['embedding_dimension'])
                    col4.metric("Alpha", stats['alpha'])
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure Ollama is running: `ollama serve`")

# -----------------------------------------------------------------------------
# TAB 2: Search
# -----------------------------------------------------------------------------
with tab2:
    st.header("🔍 Semantic Search")
    st.write("Search your knowledge base using hybrid retrieval")
    
    if not st.session_state.documents_loaded:
        st.warning("Please index documents first")
    else:
        search_query = st.text_input(
            "Search query",
            placeholder="How does hybrid search work?",
            key="search_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            num_results = st.number_input("Results", min_value=1, max_value=10, value=5)
        
        if st.button("Search", type="primary") or search_query:
            if search_query:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    results = st.session_state.rag_pipeline.search(search_query, k=num_results)
                    search_time = time.time() - start_time
                
                st.success(f"Found {len(results)} results in {search_time*1000:.0f}ms")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} · Score {result['hybrid_score']:.4f}", expanded=(i<=3)):
                        st.markdown(f'<div class="source-doc">{result["text"]}</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Hybrid", f"{result['hybrid_score']:.4f}")
                        if 'vector_score' in result:
                            col2.metric("Vector", f"{result.get('vector_score', 0):.4f}")
                        if 'bm25_score' in result:
                            col3.metric("BM25", f"{result.get('bm25_score', 0):.4f}")
            else:
                st.info("Enter a search query above")

# -----------------------------------------------------------------------------
# TAB 3: RAG
# -----------------------------------------------------------------------------
with tab3:
    st.header("💬 Ask Questions")
    st.write("Get answers grounded in your documents")
    
    if not st.session_state.documents_loaded:
        st.warning("Please index documents first")
    else:
        question = st.text_input(
            "Your question",
            placeholder="What is VectorFlow-RAG?",
            key="question_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            show_sources = st.checkbox("Show sources", value=True)
        with col2:
            show_metrics = st.checkbox("Show metrics", value=True)
        
        if st.button("Ask", type="primary") or question:
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        response = st.session_state.rag_pipeline.ask(
                            question=question,
                            k_docs=k_docs,
                            return_sources=True,
                            verbose=False
                        )
                        
                        st.subheader("Answer")
                        st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)
                        
                        if show_metrics:
                            st.subheader("Performance")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Retrieval", f"{response['metrics']['retrieval_time_ms']:.0f}ms")
                            col2.metric("Generation", f"{response['metrics']['generation_time_ms']:.0f}ms")
                            col3.metric("Total", f"{response['metrics']['total_time_ms']:.0f}ms")
                            col4.metric("Context Docs", response['metrics']['num_context_docs'])
                        
                        if show_sources and 'sources' in response:
                            st.subheader("Sources")
                            for i, source in enumerate(response['sources'], 1):
                                with st.expander(f"Source {i} · Score {source['hybrid_score']:.4f}"):
                                    st.markdown(f'<div class="source-doc">{source["text"]}</div>', unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": response["answer"],
                            "time": time.strftime("%H:%M:%S")
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Make sure Ollama is running: `ollama serve`")
            else:
                st.info("Enter a question above")
        
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("Recent Questions")
            for chat in reversed(st.session_state.chat_history[-5:]):
                with st.expander(f"[{chat['time']}] {chat['question'][:60]}..."):
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer'][:250]}...")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #475569; padding: 2rem;'>
    <p><strong>VectorFlow-RAG</strong> · Production-grade RAG framework</p>
    <p>Built for ML engineers and researchers · Open source & free</p>
</div>
""", unsafe_allow_html=True)
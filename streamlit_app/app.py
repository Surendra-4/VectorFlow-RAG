# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\streamlit_app\app.py

"""
VectorFlow-RAG Interactive Demo
Streamlit web interface for semantic search and RAG
Enhanced with professional UI/UX
"""
import streamlit as st
import sys, os
import requests

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline
import time

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="VectorFlow-RAG | Semantic Search & RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/vectorflow-rag',
        'Report a bug': 'https://github.com/yourusername/vectorflow-rag/issues',
        'About': '# VectorFlow-RAG\nProduction-Grade Semantic Search & RAG System'
    }
)

# =============================================================================
# CUSTOM CSS - PROFESSIONAL STYLING
# =============================================================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Root variables */
    :root {
        --primary-color: #0066ff;
        --secondary-color: #00d4ff;
        --accent-color: #ff6b6b;
        --success-color: #00d084;
        --warning-color: #ffa500;
        --dark-bg: #0f1419;
        --light-bg: #ffffff;
        --card-bg: #f8f9fa;
        --border-color: #e0e0e0;
        --text-primary: #1a1a1a;
        --text-secondary: #666666;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #0066ff 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #666666;
        margin: 0 0 2rem 0;
        font-size: 1.2rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #0066ff, transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0.75rem 0;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 16px rgba(0, 102, 255, 0.15);
        transform: translateY(-2px);
        border-color: #0066ff;
    }
    
    .source-doc {
        background: linear-gradient(135deg, #e3f2fd 0%, #f0f4ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0066ff;
        border: 1px solid #cfe3ff;
        margin: 0.75rem 0;
        line-height: 1.7;
        color: #333;
        box-shadow: 0 2px 8px rgba(0, 102, 255, 0.08);
    }
    
    .answer-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8feff 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #0066ff;
        margin: 1.5rem 0;
        line-height: 1.8;
        color: #1a1a1a;
        box-shadow: 0 8px 24px rgba(0, 102, 255, 0.12);
        font-size: 1.05rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #f0fdf4 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00d084;
        border: 1px solid #a6f4c5;
        color: #166534;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fff7ed 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ffa500;
        border: 1px solid #fecaca;
        color: #7c2d12;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0066ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 102, 255, 0.3);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0066ff;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 1.5rem;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        transition: all 0.3s;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        color: #0066ff;
        border-bottom: 3px solid #0066ff;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        transition: all 0.3s;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #0066ff;
        box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.1);
    }
    
    /* Expander */
    .stExpander > div > button {
        font-weight: 600;
        color: #0066ff;
        transition: all 0.3s;
    }
    
    .stExpander > div > button:hover {
        background-color: rgba(0, 102, 255, 0.05);
    }
    
    /* Checkbox */
    .stCheckbox > label {
        font-weight: 500;
        color: #333;
    }
    
    /* Status indicators */
    .status-online {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #00d084;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #ff6b6b;
        margin-right: 0.5rem;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    /* Header section */
    .header-section {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Results */
    .result-item {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s;
    }
    
    .result-item:hover {
        box-shadow: 0 8px 24px rgba(0, 102, 255, 0.15);
        border-color: #0066ff;
        transform: translateY(-2px);
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-primary {
        background-color: rgba(0, 102, 255, 0.15);
        color: #0066ff;
    }
    
    .badge-success {
        background-color: rgba(0, 208, 132, 0.15);
        color: #00d084;
    }
    
    .badge-warning {
        background-color: rgba(255, 165, 0, 0.15);
        color: #ff8c00;
    }
    
    /* Sections */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #0066ff;
        display: inline-block;
    }
    
    .section-subtitle {
        font-size: 1rem;
        color: #666;
        margin: 0.5rem 0 1.5rem 0;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer-content {
        text-align: center;
        color: #666;
        padding: 3rem 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
    }
    
    .footer-content p {
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .footer-content strong {
        color: #0066ff;
        font-weight: 700;
    }
    
    /* Loading animation */
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Utility classes */
    .text-center {
        text-align: center;
    }
    
    .text-muted {
        color: #999;
    }
    
    .mt-2 {
        margin-top: 1rem;
    }
    
    .mb-2 {
        margin-bottom: 1rem;
    }
    
    .gap-1 {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'ollama_status' not in st.session_state:
    st.session_state.ollama_status = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def check_ollama_status():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_status_badge(status):
    """Return HTML for status badge"""
    if status:
        return '<span class="badge badge-success">✓ Online</span>'
    else:
        return '<span class="badge badge-warning">✗ Offline</span>'

# =============================================================================
# HEADER SECTION
# =============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">🔍 VectorFlow-RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production-Grade Semantic Search & Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - CONFIGURATION & STATUS
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    with st.expander("🤖 Model Settings", expanded=True):
        llm_model = st.selectbox(
            "LLM Model",
            ["tinyllama", "llama3.2:1b", "mistral:7b"],
            help="Select Ollama model for text generation. Ensure it's downloaded before use."
        )
        
        alpha = st.slider(
            "Hybrid Search Alpha",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Balance between BM25 (0.0) and Vector (1.0) search. 0.5 = Balanced"
        )
        
        k_docs = st.slider(
            "Context Documents",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of documents to retrieve for RAG context"
        )
    
    st.markdown("---")
    
    with st.expander("📊 System Status", expanded=True):
        ollama_status = check_ollama_status()
        st.session_state.ollama_status = ollama_status
        
        status_html = f'<p><span class="status-{"online" if ollama_status else "offline"}"></span>Ollama: {"Connected" if ollama_status else "Disconnected"}</p>'
        st.markdown(status_html, unsafe_allow_html=True)
        
        if st.session_state.documents_loaded:
            st.markdown('<p style="color: #00d084; font-weight: 600;">✓ Documents Loaded</p>', unsafe_allow_html=True)
            if st.session_state.rag_pipeline:
                stats = st.session_state.rag_pipeline.get_stats()
                col1, col2 = st.columns(2)
                col1.metric("📄 Documents", stats['documents_ingested'])
                col2.metric("📦 Chunks", stats['chunks_indexed'])
                col1.metric("🔢 Embedding Dim", stats['embedding_dimension'])
                col2.metric("⚙️ Alpha", f"{stats['alpha']:.2f}")
        else:
            st.markdown('<p style="color: #ff8c00; font-weight: 600;">⚠️ No Documents Loaded</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("ℹ️ About VectorFlow-RAG", expanded=False):
        st.markdown("""
        ### Production-Grade RAG Framework
        
        **Key Features:**
        - 🎯 Hybrid Search (BM25 + Vectors)
        - 🤖 Local LLM with Ollama
        - 📦 Modular Architecture
        - 🆓 100% Free & Open Source
        
        **Technologies:**
        - Sentence Transformers
        - ChromaDB
        - BM25S
        - Ollama
        - Streamlit
        
        **Perfect for:**
        - Enterprise Knowledge Base Search
        - Document Q&A Systems
        - RAG Pipeline Development
        - ML Research & Experimentation
        """)

# =============================================================================
# MAIN CONTENT - TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["📝 Document Upload", "🔍 Semantic Search", "💬 Ask Questions"])

# =============================================================================
# TAB 1: DOCUMENT UPLOAD
# =============================================================================
with tab1:
    st.markdown('<div class="section-title">📝 Upload & Index Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Add documents to build your knowledge base for semantic search and Q&A</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        use_sample = st.checkbox("📚 Use Sample Documents", value=True, help="Load pre-configured sample documents about VectorFlow-RAG")
        
        if use_sample:
            st.markdown('<div class="success-box">✓ Using built-in sample documents for demonstration</div>', unsafe_allow_html=True)
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
            st.write("Enter your documents below (one per text area):")
            num_docs = st.number_input("Number of documents", min_value=1, max_value=10, value=3)
            sample_docs = []
            for i in range(num_docs):
                doc = st.text_area(f"Document {i+1}", height=120, key=f"doc_{i}", placeholder="Enter document text here...")
                if doc:
                    sample_docs.append(doc)
    
    with col2:
        st.markdown('<div style="background: linear-gradient(135deg, #f0f9ff 0%, #eff6ff 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #bfdbfe;">', unsafe_allow_html=True)
        st.markdown('<p style="font-weight: 700; color: #0066ff; margin-bottom: 1rem;">📊 Document Preview</p>', unsafe_allow_html=True)
        if sample_docs:
            total_chars = sum(len(doc) for doc in sample_docs)
            st.metric("Documents", len(sample_docs))
            st.metric("Total Characters", f"{total_chars:,}")
            if len(sample_docs) > 0:
                st.metric("Avg Doc Length", f"{total_chars // len(sample_docs):,} chars")
        else:
            st.metric("Documents", 0)
            st.metric("Total Characters", 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🚀 Index Documents", type="primary", use_container_width=True):
        if not sample_docs or all(not doc.strip() for doc in sample_docs):
            st.markdown('<div class="warning-box">⚠️ Please provide at least one document!</div>', unsafe_allow_html=True)
        else:
            with st.spinner("🔄 Processing documents..."):
                try:
                    # Initialize pipeline
                    rag = RAGPipeline(
                        index_dir="indices\\streamlit_rag",
                        alpha=alpha,
                        llm_model=llm_model
                    )
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📄 Chunking documents...")
                    progress_bar.progress(25)
                    time.sleep(0.3)
                    
                    status_text.text("🔢 Generating embeddings...")
                    progress_bar.progress(50)
                    
                    # Ingest documents
                    rag.ingest_documents(sample_docs, reset=True)
                    
                    progress_bar.progress(75)
                    status_text.text("🔍 Building indices...")
                    time.sleep(0.3)
                    
                    progress_bar.progress(100)
                    status_text.text("✓ Complete!")
                    time.sleep(0.5)
                    
                    # Save to session state
                    st.session_state.rag_pipeline = rag
                    st.session_state.documents_loaded = True
                    
                    st.markdown('<div class="success-box">✓ Documents indexed successfully!</div>', unsafe_allow_html=True)
                    st.balloons()
                    
                    # Show statistics
                    st.markdown('<div class="section-title" style="margin-top: 2rem;">📈 Indexing Statistics</div>', unsafe_allow_html=True)
                    stats = rag.get_stats()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Documents", stats['documents_ingested'], delta="Documents loaded", delta_color="off")
                    col2.metric("Chunks", stats['chunks_indexed'], delta="Chunks created", delta_color="off")
                    col3.metric("Embedding Dim", stats['embedding_dimension'], delta="Dimensions", delta_color="off")
                    col4.metric("Alpha", f"{stats['alpha']:.2f}", delta="Search balance", delta_color="off")
                    
                except Exception as e:
                    st.markdown(f'<div class="warning-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">💡 Make sure Ollama is running: `ollama serve`</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2: SEMANTIC SEARCH
# =============================================================================
with tab2:
    st.markdown('<div class="section-title">🔍 Semantic Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Search your knowledge base using hybrid retrieval combining BM25 keyword matching and vector similarity</div>', unsafe_allow_html=True)
    
    if not st.session_state.documents_loaded:
        st.markdown('<div class="warning-box">⚠️ Please index documents first in the "Document Upload" tab</div>', unsafe_allow_html=True)
    else:
        # Search interface
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="Example: How does hybrid search work? What are the key features?",
            key="search_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            num_results = st.number_input("Number of Results", min_value=1, max_value=10, value=5)
        with col3:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button or search_query:
            if search_query:
                with st.spinner("🔍 Searching..."):
                    start_time = time.time()
                    results = st.session_state.rag_pipeline.search(search_query, k=num_results)
                    search_time = time.time() - start_time
                
                st.markdown(f'<div class="success-box">✓ Found {len(results)} results in {search_time*1000:.1f}ms</div>', unsafe_allow_html=True)
                
                # Display results
                for i, result in enumerate(results, 1):
                    with st.expander(
                        f"📌 Result {i} — Score: {result['hybrid_score']:.4f}",
                        expanded=(i <= 2)
                    ):
                        # Score badges
                        score_html = f'<div style="margin-bottom: 1rem;">'
                        score_html += f'<span class="badge badge-primary">Hybrid: {result["hybrid_score"]:.4f}</span>'
                        if 'vector_score' in result:
                            score_html += f'<span class="badge badge-success">Vector: {result["vector_score"]:.4f}</span>'
                        if 'bm25_score' in result:
                            score_html += f'<span class="badge badge-warning">BM25: {result["bm25_score"]:.4f}</span>'
                        score_html += '</div>'
                        st.markdown(score_html, unsafe_allow_html=True)
                        
                        # Content
                        st.markdown(f'<div class="source-doc">{result["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">👆 Enter a search query above to get started</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 3: ASK QUESTIONS (RAG)
# =============================================================================
with tab3:
    st.markdown('<div class="section-title">💬 Ask Questions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Ask questions about your documents and receive grounded, context-aware answers powered by local LLM inference</div>', unsafe_allow_html=True)
    
    if not st.session_state.documents_loaded:
        st.markdown('<div class="warning-box">⚠️ Please index documents first in the "Document Upload" tab</div>', unsafe_allow_html=True)
    else:
        # Question input
        question = st.text_input(
            "Ask your question:",
            placeholder="Example: What is VectorFlow-RAG? How does it work?",
            key="question_input"
        )
        
        # Options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_sources = st.checkbox("📚 Show source documents", value=True, help="Display the documents used to generate the answer")
        with col2:
            show_metrics = st.checkbox("📊 Show metrics", value=True, help="Display performance metrics")
        with col3:
            show_history = st.checkbox("📜 Show history", value=True, help="Display recent questions")
        
        ask_button = st.button("💬 Ask Question", type="primary", use_container_width=True)
        
        if ask_button or question:
            if question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.rag_pipeline.ask(
                            question=question,
                            k_docs=k_docs,
                            return_sources=True,
                            verbose=False
                        )
                        
                        # Display answer
                        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
                        st.markdown('<p style="font-size: 1.2rem; font-weight: 700; color: #0066ff; margin-bottom: 1rem;">💡 Answer</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)
                        
                        # Performance metrics
                        if show_metrics:
                            st.markdown('<p style="font-size: 1.2rem; font-weight: 700; color: #0066ff; margin: 2rem 0 1rem 0;">📊 Performance Metrics</p>', unsafe_allow_html=True)
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric(
                                "Retrieval",
                                f"{response['metrics']['retrieval_time_ms']:.1f}ms",
                                help="Time to retrieve relevant documents"
                            )
                            col2.metric(
                                "Generation",
                                f"{response['metrics']['generation_time_ms']:.1f}ms",
                                help="Time to generate answer"
                            )
                            col3.metric(
                                "Total",
                                f"{response['metrics']['total_time_ms']:.1f}ms",
                                help="Total processing time"
                            )
                            col4.metric(
                                "Context Docs",
                                response['metrics']['num_context_docs'],
                                help="Number of documents used"
                            )
                        
                        # Source documents
                        if show_sources and 'sources' in response:
                            st.markdown('<p style="font-size: 1.2rem; font-weight: 700; color: #0066ff; margin: 2rem 0 1rem 0;">📚 Source Documents</p>', unsafe_allow_html=True)
                            for i, source in enumerate(response['sources'], 1):
                                with st.expander(f"📖 Source {i} — Score: {source['hybrid_score']:.4f}"):
                                    st.markdown(f'<div class="source-doc">{source["text"]}</div>', unsafe_allow_html=True)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": response["answer"],
                            "time": time.strftime("%H:%M:%S")
                        })
                        
                    except Exception as e:
                        st.markdown(f'<div class="warning-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="info-box">💡 Make sure Ollama is running: `ollama serve`</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">👆 Enter a question above to get started</div>', unsafe_allow_html=True)
        
        # Chat history
        if show_history and st.session_state.chat_history:
            st.markdown("---")
            st.markdown('<p style="font-size: 1.2rem; font-weight: 700; color: #0066ff; margin: 1.5rem 0;">📜 Recent Questions</p>', unsafe_allow_html=True)
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                with st.expander(f"🕐 [{chat['time']}] {chat['question'][:60]}..."):
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown("---")
                    st.markdown(f"**A:** {chat['answer'][:300]}...")
                    if len(chat['answer']) > 300:
                        st.caption("(truncated for preview)")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="footer-content">
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
        <strong>VectorFlow-RAG</strong> — Production-Grade Semantic Search & RAG Framework
    </p>
    <p style="font-size: 0.9rem;">
        🔗 Open Source | 🆓 100% Free | 🚀 Production-Ready | 🧠 Built with ML Excellence
    </p>
    <p style="font-size: 0.85rem; margin-top: 1rem; color: #999;">
        © 2024 • Engineered for Enterprise • Designed for Scale
    </p>
</div>
""", unsafe_allow_html=True)
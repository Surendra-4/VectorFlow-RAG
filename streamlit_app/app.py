# streamlit_app/app.py

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

# Add this at the top of app.py after imports

import os

# Check if running locally or on Streamlit Cloud
if "STREAMLIT_CLOUD" in os.environ:
    st.warning("⚠️ NOTE: Streamlit Cloud deployment requires Ollama running on your local machine or a remote server. For full functionality, run this app locally with: `ollama serve`")
else:
    # Local mode - check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            st.sidebar.error("❌ Ollama not running! Start with: ollama serve")
    except:
        st.sidebar.error("❌ Ollama not reachable at http://localhost:11434")


# Page configuration
st.set_page_config(
    page_title="VectorFlow-RAG Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-doc {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
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
st.markdown('<div class="sub-header">Production-Grade Semantic Search & RAG System</div>', unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# SIDEBAR - Configuration
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model settings
    st.subheader("Model Settings")
    
    llm_model = st.selectbox(
        "LLM Model",
        ["tinyllama", "llama3.2:1b", "mistral:7b"],
        help="Ollama model for generation. Make sure it's downloaded!"
    )
    
    alpha = st.slider(
        "Hybrid Search Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = Pure BM25, 1 = Pure Vector, 0.5 = Balanced"
    )
    
    k_docs = st.slider(
        "Context Documents",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of documents to retrieve for RAG"
    )
    
    st.markdown("---")
    
    # System info
    st.subheader("📊 System Status")
    
    if st.session_state.documents_loaded:
        st.success("✅ Documents Loaded")
        if st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_stats()
            st.metric("Documents", stats['documents_ingested'])
            st.metric("Chunks", stats['chunks_indexed'])
            st.metric("Embedding Dim", stats['embedding_dimension'])
    else:
        st.warning("⚠️ No Documents Loaded")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **VectorFlow-RAG** is a modular semantic search and RAG framework.
    
    **Features:**
    - 🎯 Hybrid Search (BM25 + Vector)
    - 🤖 Local LLM with Ollama
    - 📦 Modular Architecture
    - 🆓 100% Free & Open Source
    
    Built with: Sentence Transformers, ChromaDB, BM25S, Ollama, Streamlit
    """)

# =============================================================================
# MAIN CONTENT - Tabs
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📝 Document Upload", "🔍 Search", "💬 Ask Questions (RAG)"])

# -----------------------------------------------------------------------------
# TAB 1: Document Upload
# -----------------------------------------------------------------------------
with tab1:
    st.header("📝 Upload & Index Documents")
    st.write("Add your documents to the knowledge base for semantic search and Q&A.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sample documents option
        use_sample = st.checkbox("Use Sample Documents", value=True)
        
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
            st.write("Enter your documents below (one per text area):")
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
            st.metric("Documents", len(sample_docs))
            st.metric("Total Characters", f"{total_chars:,}")
            st.metric("Avg Doc Length", f"{total_chars // len(sample_docs):,}")
    
    # Index button
    st.markdown("---")
    if st.button("🚀 Index Documents", type="primary", use_container_width=True):
        if not sample_docs or all(not doc.strip() for doc in sample_docs):
            st.error("❌ Please provide at least one document!")
        else:
            with st.spinner("Initializing RAG pipeline..."):
                try:
                    # Initialize pipeline
                    rag = RAGPipeline(
                        index_dir="indices\\streamlit_rag",
                        alpha=alpha,
                        llm_model=llm_model
                    )
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Chunking documents...")
                    progress_bar.progress(25)
                    time.sleep(0.5)
                    
                    status_text.text("Generating embeddings...")
                    progress_bar.progress(50)
                    
                    # Ingest documents
                    rag.ingest_documents(sample_docs, reset=True)
                    
                    progress_bar.progress(75)
                    status_text.text("Building indices...")
                    time.sleep(0.5)
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    # Save to session state
                    st.session_state.rag_pipeline = rag
                    st.session_state.documents_loaded = True
                    
                    st.success("✅ Documents indexed successfully!")
                    st.balloons()
                    
                    # Show stats
                    stats = rag.get_stats()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Documents", stats['documents_ingested'])
                    col2.metric("Chunks", stats['chunks_indexed'])
                    col3.metric("Embedding Dim", stats['embedding_dimension'])
                    col4.metric("Alpha", stats['alpha'])
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure Ollama is running: `ollama serve`")

# -----------------------------------------------------------------------------
# TAB 2: Search
# -----------------------------------------------------------------------------
with tab2:
    st.header("🔍 Semantic Search")
    st.write("Search your knowledge base using hybrid retrieval (BM25 + Vector Similarity)")
    
    if not st.session_state.documents_loaded:
        st.warning("⚠️ Please index documents first (go to 'Document Upload' tab)")
    else:
        # Search input
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., How does hybrid search work?",
            key="search_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            num_results = st.number_input("Results", min_value=1, max_value=10, value=5)
        
        if st.button("🔍 Search", type="primary") or search_query:
            if search_query:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    results = st.session_state.rag_pipeline.search(search_query, k=num_results)
                    search_time = time.time() - start_time
                
                st.success(f"✅ Found {len(results)} results in {search_time*1000:.1f}ms")
                
                # Display results
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} - Score: {result['hybrid_score']:.4f}", expanded=(i<=3)):
                        st.markdown(f"**Text:**")
                        st.markdown(f'<div class="source-doc">{result["text"]}</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Hybrid Score", f"{result['hybrid_score']:.4f}")
                        if 'vector_score' in result:
                            col2.metric("Vector Score", f"{result.get('vector_score', 0):.4f}")
                        if 'bm25_score' in result:
                            col3.metric("BM25 Score", f"{result.get('bm25_score', 0):.4f}")
            else:
                st.info("👆 Enter a search query above")

# -----------------------------------------------------------------------------
# TAB 3: Ask Questions (RAG)
# -----------------------------------------------------------------------------
with tab3:
    st.header("💬 Ask Questions (RAG)")
    st.write("Ask questions and get answers grounded in your documents")
    
    if not st.session_state.documents_loaded:
        st.warning("⚠️ Please index documents first (go to 'Document Upload' tab)")
    else:
        # Question input
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What is VectorFlow-RAG?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            show_sources = st.checkbox("Show source documents", value=True)
        with col2:
            show_metrics = st.checkbox("Show performance metrics", value=True)
        
        if st.button("💬 Ask", type="primary") or question:
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
                        st.markdown("### 💡 Answer")
                        st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)
                        
                        # Show metrics
                        if show_metrics:
                            st.markdown("### 📊 Performance Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Retrieval", f"{response['metrics']['retrieval_time_ms']:.1f}ms")
                            col2.metric("Generation", f"{response['metrics']['generation_time_ms']:.1f}ms")
                            col3.metric("Total", f"{response['metrics']['total_time_ms']:.1f}ms")
                            col4.metric("Context Docs", response['metrics']['num_context_docs'])
                        
                        # Show sources
                        if show_sources and 'sources' in response:
                            st.markdown("### 📚 Source Documents")
                            for i, source in enumerate(response['sources'], 1):
                                with st.expander(f"Source {i} - Score: {source['hybrid_score']:.4f}"):
                                    st.markdown(source['text'])
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": response["answer"],
                            "time": time.strftime("%H:%M:%S")
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Make sure Ollama is running: `ollama serve`")
            else:
                st.info("👆 Enter a question above")
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📜 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                with st.expander(f"[{chat['time']}] {chat['question'][:50]}..."):
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer'][:200]}...")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>VectorFlow-RAG</strong> - Built with ❤️ for ML Engineers & Researchers</p>
    <p>🔗 Open Source | 🆓 100% Free | 🚀 Production-Ready</p>
</div>
""", unsafe_allow_html=True)

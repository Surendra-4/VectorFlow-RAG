"""
Complete RAG pipeline integrating all components
"""

import os
import time
import sys
from typing import Dict, List, Optional

from src.bm25_retriever import BM25Retriever
from src.chunker import TextChunker
from src.embedder import Embedder
from src.hybrid_retriever import HybridRetriever
from src.llm_client import OllamaClient
from src.vector_store import VectorStore

sys.path.append(os.path.dirname(__file__))


class RAGPipeline:
    """End-to-end RAG system combining retrieval and generation"""

    def __init__(
        self,
        index_dir: str = "indices\\rag_system",
        alpha: float = 0.5,
        llm_model: str = "tinyllama",
    ):
        """
        Initialize RAG pipeline

        Args:
            index_dir: Directory for storing indices
            alpha: Hybrid search weight (0=pure BM25, 1=pure vector, 0.5=balanced)
            llm_model: Ollama model name (tinyllama, llama3.2:1b, etc.)
        """
        print("=" * 70)
        print("Initializing VectorFlow-RAG Pipeline")
        print("=" * 70)

        # Initialize components
        print("\n[1/5] Loading embedding model...")
        self.embedder = Embedder()

        print("[2/5] Initializing text chunker...")
        self.chunker = TextChunker(chunk_size=500, overlap=50)

        print("[3/5] Setting up vector store...")
        self.vector_store = VectorStore(persist_directory=f"{index_dir}\\chroma")

        print("[4/5] Connecting to Ollama LLM...")
        self.llm = OllamaClient(model=llm_model)

        print("[5/5] Finalizing setup...")
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.corpus = []
        self.alpha = alpha
        self.document_count = 0

        print("\n✓ RAG Pipeline Ready!")
        print(f"  - Hybrid alpha: {self.alpha}")
        print(f"  - LLM model: {llm_model}")
        print("=" * 70 + "\n")

    def ingest_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        reset: bool = True,
    ):
        """
        Ingest and index documents into the RAG system

        Args:
            documents: List of text documents to index
            metadatas: Optional metadata for each document
            reset: Whether to reset existing indices
        """
        print(f"\n{'='*70}")
        print(f"INGESTING {len(documents)} DOCUMENTS")
        print(f"{'='*70}\n")

        # Step 1: Chunk documents
        print("[Step 1/4] Chunking documents...")
        all_chunks = []
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if metadatas else {"doc_id": i, "source": f"document_{i}"}
            chunks = self.chunker.chunk_text(doc, metadata=metadata)
            all_chunks.extend(chunks)
            print(f"  ✓ Document {i+1}: {len(chunks)} chunks created")

        print(f"\n  Total chunks created: {len(all_chunks)}")

        # Extract text and metadata
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        chunk_metas = [chunk["metadata"] for chunk in all_chunks]

        # Step 2: Generate embeddings
        print("\n[Step 2/4] Generating embeddings...")
        embeddings = self.embedder.encode(chunk_texts, show_progress=True)
        print(f"  ✓ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

        # Step 3: Index in vector store
        print("\n[Step 3/4] Building vector index...")
        if reset:
            try:
                self.vector_store.delete_collection()
                print("  ✓ Old collection deleted successfully")
            except Exception as e:
                print(f"  (warning) could not delete collection: {e}")

        # Recreate a clean VectorStore and collection
        self.vector_store = VectorStore(persist_directory=self.vector_store.persist_directory)

        self.vector_store.add_documents(
            texts=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=chunk_metas,
            ids=[f"chunk_{i}" for i in range(len(chunk_texts))],
        )
        print(f"  ✓ Vector index built with {len(chunk_texts)} chunks")

        # Step 4: Build BM25 index
        print("\n[Step 4/4] Building BM25 lexical index...")
        self.bm25_retriever = BM25Retriever(corpus=chunk_texts)
        self.corpus = chunk_texts
        print("  ✓ BM25 index built")

        # Create hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            alpha=self.alpha,
        )

        self.document_count = len(documents)

        print(f"\n{'='*70}")
        print("✓ INGESTION COMPLETE!")
        print(f"  - Documents: {len(documents)}")
        print(f"  - Chunks: {len(chunk_texts)}")
        print("  - Ready for search and Q&A")
        print(f"{'='*70}\n")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant documents (retrieval only, no generation)

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant documents with scores
        """
        if self.hybrid_retriever is None:
            raise ValueError("❌ No documents ingested! Call ingest_documents() first.")

        print(f"\n🔍 Searching for: '{query}'")
        results = self.hybrid_retriever.search(query, k=k)
        print(f"✓ Found {len(results)} results\n")
        return results

    def ask(
        self,
        question: str,
        k_docs: int = 3,
        return_sources: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        RAG: Retrieve relevant documents and generate answer

        Args:
            question: User question
            k_docs: Number of context documents to retrieve
            return_sources: Whether to return source documents
            verbose: Print detailed progress

        Returns:
            Dict with answer, sources, and performance metrics
        """
        start_time = time.time()

        if verbose:
            print(f"\n{'='*70}")
            print("RAG QUERY")
            print(f"{'='*70}")
            print(f"Question: {question}\n")

        # Step 1: Retrieve relevant documents
        if verbose:
            print("[1/2] Retrieving relevant context...")

        results = self.search(question, k=k_docs)
        retrieval_time = time.time() - start_time

        if verbose:
            print("Retrieved documents:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. (score: {r['hybrid_score']:.3f}) {r['text'][:80]}...")

        # Extract context
        context_docs = [r["text"] for r in results]

        # Step 2: Generate answer
        if verbose:
            print("\n[2/2] Generating answer with LLM...")

        gen_start = time.time()
        answer = self.llm.generate(
            prompt=question, context=context_docs, max_tokens=512, temperature=0.7
        )
        generation_time = time.time() - gen_start
        total_time = time.time() - start_time

        if verbose:
            print(f"\n✓ Answer generated in {generation_time:.2f}s")
            print(f"{'='*70}\n")

        # Build response
        response = {
            "question": question,
            "answer": answer,
            "metrics": {
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "generation_time_ms": round(generation_time * 1000, 2),
                "total_time_ms": round(total_time * 1000, 2),
                "num_context_docs": len(context_docs),
                "alpha": self.alpha,
            },
        }

        if return_sources:
            response["sources"] = results

        return response

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "documents_ingested": self.document_count,
            "chunks_indexed": len(self.corpus),
            "embedding_dimension": self.embedder.dimension,
            "alpha": self.alpha,
            "model": self.embedder.model_name,
        }


# =============================================================================
# DEMO: Test the complete pipeline
# =============================================================================


def demo():
    """Run a complete demo of VectorFlow-RAG"""

    print("\n" + "=" * 70)
    print(" VectorFlow-RAG DEMO ".center(70, "="))
    print("=" * 70)

    # Sample knowledge base about VectorFlow-RAG
    documents = [
        (
            "VectorFlow-RAG is a production-grade, modular semantic retrieval and RAG framework. "
            "It is designed for machine learning engineers and researchers who want full transparency and control over their search pipeline. "
            "Unlike black-box solutions like LangChain or ElasticSearch, VectorFlow exposes every internal layer: embedding, indexing, ranking, and generation. "
            "The system is built to be reproducible, swappable, and benchmarkable."
        ),
        (
            "The embedding layer uses state-of-the-art models from sentence-transformers. "
            "By default, it uses all-MiniLM-L6-v2, a lightweight 80MB model with 384 dimensions. "
            "These models convert text into high-dimensional vectors that capture semantic meaning. "
            "The embeddings are then stored in efficient vector databases like ChromaDB and FAISS. "
            "Users can easily swap embedding models through YAML configuration files."
        ),
        (
            "Hybrid search is the core retrieval strategy in VectorFlow-RAG. "
            "It combines BM25 lexical matching with vector similarity search. "
            "BM25 is excellent at exact keyword matches and handles rare terms well, while vector search captures semantic similarity and understands context. "
            "The fusion parameter alpha controls the balance between the two approaches, where alpha=0.5 gives equal weight to both methods."
        ),
        (
            "For text generation, VectorFlow-RAG uses Ollama to run large language models locally. "
            "This approach ensures complete privacy and eliminates API costs. "
            "Popular models include TinyLlama (600MB), Llama 3.2 (1.3GB), and Mistral (4GB), which can all run on consumer hardware. "
            "The system supports any Ollama model through simple configuration changes. "
            "Generation is combined with retrieved context to produce grounded, factual answers."
        ),
        (
            "The framework includes comprehensive experiment tracking using MLflow and DagsHub. "
            "All experiments are automatically logged with metrics like NDCG, MRR, recall@k, and latency. "
            "This makes VectorFlow-RAG suitable for research applications and systematic ablation studies. "
            "Users can compare different embedding models, chunking strategies, and alpha parameters. "
            "The system is fully reproducible with versioned configurations and deterministic results."
        ),
        (
            "VectorFlow-RAG is built with modern ML engineering practices. "
            "It uses FastAPI for REST API endpoints, Docker for containerization, and GitHub Actions for CI/CD. "
            "The system can be deployed to free platforms like Streamlit Community Cloud, Render, or Hugging Face Spaces. "
            "All components are modular and can be replaced independently. "
            "The codebase includes comprehensive tests and follows production-ready design patterns."
        ),
    ]

    # Initialize pipeline
    rag = RAGPipeline(
        index_dir="indices\\demo_rag",
        alpha=0.5,  # Balanced hybrid search
        llm_model="tinyllama",
    )

    # Ingest documents
    rag.ingest_documents(documents)

    # Display stats
    print("\nPipeline Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # =============================================================================
    # TEST 1: Search Only (No Generation)
    # =============================================================================
    print("\n\n" + "=" * 70)
    print(" TEST 1: SEARCH ONLY ".center(70, "="))
    print("=" * 70)

    search_query = "What embedding models does VectorFlow use?"
    search_results = rag.search(search_query, k=3)

    print("Top 3 Results:")
    for i, result in enumerate(search_results, 1):
        print(f"\n{i}. Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"   Text: {result['text'][:150]}...")

    # =============================================================================
    # TEST 2: RAG (Retrieval + Generation)
    # =============================================================================
    print("\n\n" + "=" * 70)
    print(" TEST 2: RAG (RETRIEVAL + GENERATION) ".center(70, "="))
    print("=" * 70)

    questions = [
        "What is VectorFlow-RAG and what makes it different?",
        "How does hybrid search work in this system?",
        "What LLM options are available?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n\n--- Question {i} ---")
        response = rag.ask(question, k_docs=2, verbose=True)

        print("ANSWER:")
        print("-" * 70)
        print(response["answer"])
        print("-" * 70)

        print("\nPERFORMANCE METRICS:")
        for metric, value in response["metrics"].items():
            print(f"  {metric}: {value}")

        print("\nSOURCE DOCUMENTS:")
        for j, source in enumerate(response["sources"], 1):
            print(f"  {j}. (score: {source['hybrid_score']:.3f}) {source['text'][:80]}...")

    print("\n\n" + "=" * 70)
    print(" DEMO COMPLETE! ".center(70, "="))
    print("=" * 70)
    print("\n✓ VectorFlow-RAG is working perfectly!")
    print("✓ You can now use this for your own documents")
    print("✓ Next: Build a Streamlit UI for interactive demos\n")


if __name__ == "__main__":
    demo()

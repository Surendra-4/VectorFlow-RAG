"""
Pytest configuration and fixtures for VectorFlow-RAG tests
"""
import os
import sys
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="bm25s")


# ============================================================================
# GLOBAL TEST CONFIGURATION
# ============================================================================

# Set environment variables ONCE at pytest startup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Reproducible random seed for all tests
RANDOM_SEED = 42

import pytest


def pytest_configure(config):
    """Configure pytest at startup - runs once per test session"""
    np.random.seed(RANDOM_SEED)
    
    # Register custom test markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (skip in CI)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance/benchmark tests"
    )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def random_seed():
    """Provide the random seed used for all tests"""
    return RANDOM_SEED


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test artifacts.
    
    Automatically cleaned up after test completion.
    Uses pathlib for cross-platform compatibility.
    
    Example:
        def test_something(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("content")
    """
    temp_path = Path(tempfile.mkdtemp(prefix="vectorflow_test_"))
    yield temp_path
    
    # Guaranteed cleanup even if test fails
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path)
        except Exception as e:
            print(f"Warning: Failed to remove {temp_path}: {e}")


@pytest.fixture
def temp_chroma_dir(temp_dir):
    """Create a temporary directory for ChromaDB indices"""
    chroma_dir = temp_dir / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return str(chroma_dir)


@pytest.fixture
def embedding_data(random_seed):
    """
    Provide consistent test embeddings with reproducible randomness.
    
    Returns:
        Tuple of (texts, embeddings) with fixed random seed
    """
    np.random.seed(random_seed)
    texts = ["Document 1", "Document 2", "Document 3"]
    embeddings = [np.random.random(384) for _ in texts]
    return texts, embeddings


@pytest.fixture
def mock_ollama_env(monkeypatch):
    """
    Configure environment to skip/mock Ollama in CI environments.
    
    Use this fixture in tests that would normally use OllamaClient.
    """
    monkeypatch.setenv("MOCK_OLLAMA", "true")
    return None
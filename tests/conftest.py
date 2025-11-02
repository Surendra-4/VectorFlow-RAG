"""
Pytest configuration and shared fixtures
"""

import os
import sys
import pytest
from sentence_transformers import SentenceTransformer

# ---------- Prevent Windows thread crashes ----------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DISABLE_TQDM"] = "1"

# ---------- Add src to path ----------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def shared_embedder():
    """Load model once for all tests (CPU only, thread-safe)"""
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    model.max_seq_length = 256
    return model

@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory"""
    test_dir = "indices/test_data"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir

@pytest.fixture
def cleanup_indices():
    """Clean up test indices after tests"""
    yield

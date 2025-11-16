# tests/conftest.py

"""
Pytest configuration and fixtures for VectorFlow-RAG tests.

Centralizes pytest setup, environment configuration, and shared fixtures.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION & MARKER REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════


def pytest_configure(config):
    """Register custom markers and configure pytest."""
    # Register custom markers so pytest doesn't warn about unknown markers
    config.addinivalue_line("markers", "integration: marks tests as integration (require external services like Ollama)")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "slow: marks tests as slow")


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables and configuration."""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Set environment variables for CI/CD and test environments
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Signal that we're in a test environment
    os.environ["PYTEST_RUNNING"] = "true"

    # If running in GitHub Actions CI, use mocking
    if os.environ.get("GITHUB_ACTIONS"):
        os.environ["MOCK_OLLAMA"] = "true"

    yield

    # Cleanup after all tests


# ═══════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_dir():
    """
    Provide a temporary directory that's automatically cleaned up.

    Yields:
        Path object pointing to a temporary directory

    Automatically removes the directory after the test completes.
    """
    # Create temporary directory
    temp_path = Path(tempfile.mkdtemp(prefix="vectorflow_test_"))

    yield temp_path

    # Cleanup: Remove directory after test
    def cleanup_with_retry(path, retries=5):
        """Retry-safe deletion to handle Windows file locks."""
        for attempt in range(retries):
            try:
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
                return
            except PermissionError:
                import time

                time.sleep(0.5)
        # If all retries fail, at least try ignore_errors
        shutil.rmtree(path, ignore_errors=True)

    cleanup_with_retry(temp_path)


@pytest.fixture
def embedding_data():
    """
    Provide reproducible embedding data for testing.

    Returns:
        dict with test embeddings and metadata
    """
    np.random.seed(42)  # Ensure reproducibility

    embeddings = {
        "query": np.random.random(384),  # Single query embedding
        "documents": [np.random.random(384) for _ in range(5)],  # 5 document embeddings
        "dimension": 384,
        "model": "all-MiniLM-L6-v2",
    }

    return embeddings


@pytest.fixture
def random_seed():
    """
    Set random seed for deterministic tests.

    Ensures all tests produce the same random numbers when rerun.
    """
    np.random.seed(42)
    return 42


@pytest.fixture
def mock_ollama_env():
    """
    Set up mock Ollama environment for testing.

    Signals to OllamaClient to use mocks instead of real connections.
    """
    original_mock = os.environ.get("MOCK_OLLAMA")
    os.environ["MOCK_OLLAMA"] = "true"

    yield

    # Restore original value
    if original_mock is None:
        os.environ.pop("MOCK_OLLAMA", None)
    else:
        os.environ["MOCK_OLLAMA"] = original_mock


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST HOOKS FOR BETTER REPORTING
# ═══════════════════════════════════════════════════════════════════════════


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to automatically skip integration tests in CI.

    Also adds markers based on test names if not already marked.
    """
    # If running in CI (GitHub Actions), skip integration tests
    if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
        skip_integration = pytest.mark.skip(reason="Skipping integration tests in CI")
        for item in items:
            if "integration" in item.keywords or "TestErrorHandling" in item.nodeid:
                # Don't skip if explicitly marked as integration
                if "integration" not in item.keywords:
                    continue
                # Skip only if it requires Ollama
                if any(name in item.nodeid for name in ["ask", "rag_pipeline", "llm"]):
                    item.add_marker(skip_integration)


def pytest_runtest_logreport(report):
    """Add custom reporting for test results."""
    if report.when == "call":
        if hasattr(report, "wasxfail"):
            pass  # Expected failure, don't log


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION (pyproject.toml replacement)
# ═══════════════════════════════════════════════════════════════════════════

pytest_plugins = []  # Can add plugins here if needed

# These settings would normally be in pyproject.toml or pytest.ini
# but are defined here for completeness:

PYTEST_CONFIG = {
    "testpaths": ["tests"],
    "python_files": ["test_*.py"],
    "python_classes": ["Test*"],
    "python_functions": ["test_*"],
    "addopts": "-v --strict-markers --tb=short --disable-warnings -ra",
    "markers": [
        "slow: marks tests as slow",
        "integration: marks tests as integration (require Ollama)",
        "unit: marks tests as unit tests",
        "performance: marks performance tests",
        "benchmark: marks benchmarking tests",
    ],
    "timeout": 300,
}

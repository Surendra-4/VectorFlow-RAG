"""
Pytest configuration and shared fixtures
"""

import os
import sys

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory"""
    test_dir = "indices/test_data"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Cleanup is handled by pytest


@pytest.fixture
def cleanup_indices():
    """Clean up test indices after tests"""
    yield
    # Cleanup could go here
    pass

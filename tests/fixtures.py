"""
Shared fixtures for tests
"""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def sample_corpus():
    """Session-level fixture: sample corpus"""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps understand text",
        "Computer vision enables image recognition",
        "Reinforcement learning trains agents with rewards",
        "Supervised learning requires labeled data",
        "Unsupervised learning finds patterns in data",
        "Transfer learning reuses pre-trained models",
    ]


@pytest.fixture
def sample_embeddings():
    """Function-level fixture: random embeddings"""
    return np.random.random((8, 384))


@pytest.fixture
def sample_queries():
    """Function-level fixture: sample queries"""
    return [
        "What is machine learning?",
        "How does neural network work?",
        "Explain NLP",
        "Image recognition methods",
        "Reinforcement learning basics",
    ]

# tests/test_faiss_store.py

"""
Unit tests for the FAISS-backed vector store.

Covers CRUD, multiple index types, persistence/reload, deletion, error
paths, and protocol conformance. Recall and latency comparisons against
ChromaDB live in tests/test_vector_store_parity.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import VectorStoreSettings, reset_settings_cache
from src.faiss_store import FAISSVectorStore
from src.interfaces import VectorStoreProtocol
from src.vector_store import make_vector_store


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _random_embeddings(n: int, dim: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return _normalize(rng.randn(n, dim).astype(np.float32))


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #


class TestFAISSInitialization:
    def test_creates_persist_directory(self, tmp_path):
        target = tmp_path / "faiss_init"
        fs = FAISSVectorStore(persist_directory=str(target))
        assert target.exists()
        assert Path(fs.persist_directory) == target

    def test_default_index_type_is_hnsw(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        assert fs.index_type == "hnsw"

    def test_protocol_conformance(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        assert isinstance(fs, VectorStoreProtocol)

    def test_explicit_kwargs_override_settings(self, tmp_path):
        fs = FAISSVectorStore(
            persist_directory=str(tmp_path),
            index_type="flat",
            hnsw_m=8,
            hnsw_ef_construction=64,
            hnsw_ef_search=16,
        )
        assert fs.index_type == "flat"
        assert fs.hnsw_m == 8
        assert fs.hnsw_ef_construction == 64
        assert fs.hnsw_ef_search == 16

    def test_collection_name_default_from_settings(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        assert fs.collection_name == "vectorflow_docs"

    def test_windows_path_normalized(self, tmp_path):
        target = tmp_path / "win_path"
        fs = FAISSVectorStore(persist_directory=f"{tmp_path}\\win_path")
        # Either native or cleaned form should produce the same target.
        assert Path(fs.persist_directory).resolve() == target.resolve()


# --------------------------------------------------------------------------- #
# Document add / search
# --------------------------------------------------------------------------- #


class TestFAISSDocumentOps:
    @pytest.fixture
    def store(self, tmp_path):
        return FAISSVectorStore(persist_directory=str(tmp_path / "faiss_ops"))

    def test_add_single_document(self, store):
        embs = _random_embeddings(1)
        store.add_documents(["hello"], embs.tolist())
        assert store.get_stats()["total_documents"] == 1

    def test_add_multiple(self, store):
        embs = _random_embeddings(5)
        store.add_documents([f"d{i}" for i in range(5)], embs.tolist())
        assert store.get_stats()["total_documents"] == 5

    def test_add_with_explicit_ids_and_metadata(self, store):
        embs = _random_embeddings(3)
        store.add_documents(
            texts=["a", "b", "c"],
            embeddings=embs.tolist(),
            metadatas=[{"src": "x"}, {"src": "y"}, {"src": "z"}],
            ids=["doc-1", "doc-2", "doc-3"],
        )
        assert store.ids == ["doc-1", "doc-2", "doc-3"]
        assert store.metadatas[0]["src"] == "x"

    def test_auto_id_generation(self, store):
        embs = _random_embeddings(2)
        store.add_documents(["foo", "bar"], embs.tolist())
        assert store.ids[0].startswith("id_")
        assert len(set(store.ids)) == 2

    def test_default_metadata_when_omitted(self, store):
        embs = _random_embeddings(2)
        store.add_documents(["foo", "bar"], embs.tolist())
        assert all(m == {"source": "manual"} for m in store.metadatas)

    def test_numpy_array_inputs_accepted(self, store):
        embs = _random_embeddings(3)
        store.add_documents(["a", "b", "c"], embs)  # raw numpy, not list
        assert store.get_stats()["total_documents"] == 3

    def test_dim_mismatch_raises(self, store):
        store.add_documents(["a"], _random_embeddings(1, dim=384).tolist())
        with pytest.raises(ValueError):
            store.add_documents(["b"], _random_embeddings(1, dim=256).tolist())

    def test_length_mismatch_raises(self, store):
        embs = _random_embeddings(3)
        with pytest.raises(ValueError):
            store.add_documents(texts=["a", "b"], embeddings=embs.tolist())

    def test_search_returns_expected_shape(self, store):
        embs = _random_embeddings(5)
        store.add_documents([f"d{i}" for i in range(5)], embs.tolist())
        res = store.search(embs[0], n_results=3)
        # Phase 3.5+: ids field added alongside the original three.
        assert {"documents", "distances", "metadatas", "ids"}.issubset(res.keys())
        assert len(res["documents"]) == 3
        assert len(res["distances"]) == 3
        assert len(res["metadatas"]) == 3
        assert len(res["ids"]) == 3

    def test_search_returns_self_first(self, store):
        # A normalized vector queried against itself in a Flat index
        # should return itself at rank 1 with cosine similarity 1.
        store.index_type = "flat"
        embs = _random_embeddings(8)
        store.add_documents([f"doc_{i}" for i in range(8)], embs.tolist())
        res = store.search(embs[3], n_results=1)
        assert res["documents"][0] == "doc_3"
        # Distance = 1 - cos_sim → close to 0 for self-match.
        assert res["distances"][0] == pytest.approx(0.0, abs=1e-3)

    def test_search_empty_store_returns_empty(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path / "empty"))
        res = fs.search(np.zeros(384, dtype=np.float32), n_results=5)
        assert res["documents"] == []
        assert res["distances"] == []
        assert res["metadatas"] == []
        assert res["ids"] == []

    def test_search_n_results_clamped_to_corpus_size(self, store):
        embs = _random_embeddings(3)
        store.add_documents(["a", "b", "c"], embs.tolist())
        res = store.search(embs[0], n_results=100)
        assert len(res["documents"]) == 3


# --------------------------------------------------------------------------- #
# Persistence / reload
# --------------------------------------------------------------------------- #


class TestFAISSPersistence:
    def test_index_and_sidecar_written(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        embs = _random_embeddings(4)
        fs.add_documents(["a", "b", "c", "d"], embs.tolist())
        assert (tmp_path / f"{fs.collection_name}.faiss").exists()
        assert (tmp_path / f"{fs.collection_name}.json").exists()

    def test_reload_restores_state(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        embs = _random_embeddings(6)
        fs.add_documents([f"d{i}" for i in range(6)], embs.tolist(), ids=[f"x{i}" for i in range(6)])

        del fs
        fs2 = FAISSVectorStore(persist_directory=str(tmp_path))
        assert len(fs2) == 6
        assert fs2.ids == [f"x{i}" for i in range(6)]
        assert fs2.documents == [f"d{i}" for i in range(6)]

    def test_reload_search_consistent(self, tmp_path):
        fs1 = FAISSVectorStore(persist_directory=str(tmp_path))
        embs = _random_embeddings(20)
        fs1.add_documents([f"d{i}" for i in range(20)], embs.tolist())
        before = fs1.search(embs[5], n_results=5)["documents"]

        fs2 = FAISSVectorStore(persist_directory=str(tmp_path))
        after = fs2.search(embs[5], n_results=5)["documents"]
        assert before == after

    def test_corrupted_sidecar_fresh_start(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        fs.add_documents(["a"], _random_embeddings(1).tolist())
        # Truncate sidecar to corrupt it.
        sidecar = tmp_path / f"{fs.collection_name}.json"
        sidecar.write_text("{not json")
        fs2 = FAISSVectorStore(persist_directory=str(tmp_path))
        # Should fall through to fresh-start state, not crash.
        assert len(fs2) == 0


# --------------------------------------------------------------------------- #
# Deletion / reset
# --------------------------------------------------------------------------- #


class TestFAISSDeletion:
    def test_delete_clears_in_memory(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        fs.add_documents(["a", "b"], _random_embeddings(2).tolist())
        fs.delete_collection()
        assert len(fs) == 0
        assert fs.index is None

    def test_delete_removes_disk_artifacts(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        fs.add_documents(["a"], _random_embeddings(1).tolist())
        fs.delete_collection()
        assert not (tmp_path / f"{fs.collection_name}.faiss").exists()
        assert not (tmp_path / f"{fs.collection_name}.json").exists()

    def test_create_collection_with_reset(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        fs.add_documents(["a", "b"], _random_embeddings(2).tolist())
        fs.create_collection(reset=True)
        assert len(fs) == 0

    def test_search_after_delete_returns_empty(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        fs.add_documents(["a"], _random_embeddings(1).tolist())
        fs.delete_collection()
        res = fs.search(np.zeros(384, dtype=np.float32), n_results=5)
        assert res["documents"] == []


# --------------------------------------------------------------------------- #
# Multiple index types
# --------------------------------------------------------------------------- #


class TestIndexTypes:
    def test_flat_index_works(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path), index_type="flat")
        embs = _random_embeddings(10)
        fs.add_documents([f"d{i}" for i in range(10)], embs.tolist())
        res = fs.search(embs[2], n_results=3)
        assert res["documents"][0] == "d2"

    def test_hnsw_index_works(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path), index_type="hnsw")
        embs = _random_embeddings(50, seed=42)
        fs.add_documents([f"d{i}" for i in range(50)], embs.tolist())
        res = fs.search(embs[10], n_results=3)
        # Self should appear in the top-3 even at HNSW recall.
        assert "d10" in res["documents"]

    def test_ivf_index_works(self, tmp_path):
        # IVF requires more vectors to train sensibly; provide enough.
        fs = FAISSVectorStore(persist_directory=str(tmp_path), index_type="ivf")
        embs = _random_embeddings(100, seed=1)
        fs.add_documents([f"d{i}" for i in range(100)], embs.tolist())
        res = fs.search(embs[7], n_results=5)
        assert len(res["documents"]) > 0

    def test_unknown_index_type_raises(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path), index_type="bogus")
        with pytest.raises(ValueError):
            fs.add_documents(["a"], _random_embeddings(1).tolist())

    def test_set_ef_search_changes_param(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path), index_type="hnsw")
        fs.add_documents(["a", "b"], _random_embeddings(2).tolist())
        fs.set_ef_search(128)
        assert fs.hnsw_ef_search == 128
        assert fs.index.hnsw.efSearch == 128

    def test_set_ef_search_noop_for_flat(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path), index_type="flat")
        fs.add_documents(["a"], _random_embeddings(1).tolist())
        # Should not raise even though there's no .hnsw.
        fs.set_ef_search(64)


# --------------------------------------------------------------------------- #
# Factory / config integration
# --------------------------------------------------------------------------- #


class TestFactoryDispatch:
    def test_factory_returns_chroma_by_default(self, tmp_path):
        from src.vector_store import VectorStore

        store = make_vector_store(persist_directory=str(tmp_path / "f1"), backend="chromadb")
        assert isinstance(store, VectorStore)

    def test_factory_returns_faiss_when_selected(self, tmp_path):
        store = make_vector_store(persist_directory=str(tmp_path / "f2"), backend="faiss")
        assert isinstance(store, FAISSVectorStore)

    def test_factory_picks_backend_from_settings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VFR_VECTOR_STORE__BACKEND", "faiss")
        reset_settings_cache()
        try:
            store = make_vector_store(persist_directory=str(tmp_path / "f3"))
            assert isinstance(store, FAISSVectorStore)
        finally:
            reset_settings_cache()

    def test_factory_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            make_vector_store(backend="lucene")

    def test_factory_forwards_kwargs_to_faiss(self, tmp_path):
        store = make_vector_store(
            persist_directory=str(tmp_path / "f4"),
            backend="faiss",
            index_type="flat",
        )
        assert isinstance(store, FAISSVectorStore)
        assert store.index_type == "flat"


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


class TestStats:
    def test_stats_after_add(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        fs.add_documents(["a", "b", "c"], _random_embeddings(3).tolist())
        stats = fs.get_stats()
        assert stats["total_documents"] == 3
        assert stats["embedding_dim"] == 384
        assert stats["backend"] == "faiss"
        assert stats["index_type"] == "hnsw"

    def test_stats_after_delete(self, tmp_path):
        fs = FAISSVectorStore(persist_directory=str(tmp_path))
        fs.add_documents(["a"], _random_embeddings(1).tolist())
        fs.delete_collection()
        stats = fs.get_stats()
        assert stats["total_documents"] == 0

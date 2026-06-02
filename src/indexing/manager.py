# src/indexing/manager.py

"""
IndexManager (Phase 12e) — lifecycle orchestration for named vector indexes.

Owns the on-disk index directories (``indices/<name>/``) and an
:class:`IndexRegistry` of :class:`IndexProfile` metadata. Provides the full
named-entity lifecycle:

* ``create_index``  — build + populate a store, register its profile
* ``load_index``    — get a live store handle for a named index
* ``switch_index``  — set the active index and return its store
* ``delete_index``  — drop the profile and remove its directory
* ``export_index``  — archive an index (data + profile) to a ``.zip``
* ``import_index``  — restore an archived index under a new name
* ``train_index``   — train a trainable FAISS recipe (Phase 12f extends)
* ``benchmark_index`` — measure quality/latency (Phase 12i implements)

The manager is decoupled from the running pipeline: it works with already
embedded data ``(texts, embeddings, metadatas, ids)`` — exactly what
``RAGPipeline._index_chunks`` produces — so it's fully testable with synthetic
vectors and no model load.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.indexing.profile import IndexProfile
from src.indexing.registry import IndexRegistry, IndexRegistryError, validate_index_name
from src.logging_setup import get_logger

logger = get_logger(__name__)

# Fixed collection name within an index directory — keeps the on-disk file
# names deterministic regardless of the global settings.
INDEX_COLLECTION = "index"
PROFILE_ARCHIVE_NAME = "_index_profile.json"


class IndexManager:
    """Build, load, switch, delete, export and import named vector indexes."""

    def __init__(
        self,
        registry: Optional[IndexRegistry] = None,
        indices_root: Optional[Path] = None,
    ):
        if indices_root is None:
            from src.config import get_settings

            indices_root = Path(get_settings().app.project_root) / "indices" / "named"
        self.indices_root = Path(indices_root)
        self.indices_root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or IndexRegistry()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def index_dir(self, name: str) -> Path:
        validate_index_name(name)
        return self.indices_root / name

    # ------------------------------------------------------------------ #
    # Store construction
    # ------------------------------------------------------------------ #

    def _build_store(self, profile: IndexProfile, *, persist_directory: Path):
        """Construct a vector store for ``profile`` at ``persist_directory``.

        FAISS recipe support beyond hnsw/flat/ivf arrives in Phase 12f; this
        maps the profile's build/search params onto the existing store kwargs.
        """
        from src.vector_store import make_vector_store

        persist_directory.mkdir(parents=True, exist_ok=True)

        if profile.backend == "chromadb":
            return make_vector_store(
                persist_directory=str(persist_directory),
                collection_name=INDEX_COLLECTION,
                backend="chromadb",
            )
        if profile.backend == "faiss":
            bp = profile.build_params or {}
            sp = profile.search_params or {}
            kwargs: Dict[str, Any] = {"index_type": profile.index_type}
            # Map known FAISS params; unknown keys are ignored by the store.
            if "hnsw_m" in bp:
                kwargs["hnsw_m"] = bp["hnsw_m"]
            if "hnsw_ef_construction" in bp:
                kwargs["hnsw_ef_construction"] = bp["hnsw_ef_construction"]
            if "hnsw_ef_search" in sp:
                kwargs["hnsw_ef_search"] = sp["hnsw_ef_search"]
            return make_vector_store(
                persist_directory=str(persist_directory),
                collection_name=INDEX_COLLECTION,
                backend="faiss",
                **kwargs,
            )
        raise IndexRegistryError(f"Unknown backend in profile: {profile.backend!r}")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def create_index(
        self,
        profile: IndexProfile,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        *,
        make_active: bool = False,
        overwrite: bool = False,
    ) -> IndexProfile:
        """Build + populate a new named index, then register its profile."""
        validate_index_name(profile.name)
        if self.registry.exists(profile.name) and not overwrite:
            raise IndexRegistryError(f"Index {profile.name!r} already exists")

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2D; got shape {emb.shape}")

        dim = int(emb.shape[1]) if emb.size else profile.vector_dimension
        profile.vector_dimension = dim
        profile.num_vectors = len(texts)
        profile.created_at = profile.created_at or time.time()
        profile.touch()

        target = self.index_dir(profile.name)
        if target.exists() and overwrite:
            shutil.rmtree(target, ignore_errors=True)

        store = self._build_store(profile, persist_directory=target)
        if len(texts):
            store.add_documents(
                texts=list(texts),
                embeddings=emb.tolist(),
                metadatas=list(metadatas) if metadatas is not None else None,
                ids=list(ids) if ids is not None else None,
            )

        self.registry.register(profile, make_active=make_active, overwrite=overwrite)
        logger.info(
            "Created index %r (backend=%s type=%s dim=%d n=%d)",
            profile.name, profile.backend, profile.index_type, dim, profile.num_vectors,
        )
        return profile

    def load_index(self, name: str):
        """Return a live store handle for the named index (reads from disk)."""
        profile = self.registry.get(name)
        target = self.index_dir(name)
        if not target.exists():
            raise IndexRegistryError(
                f"Index directory missing for {name!r}: {target}"
            )
        store = self._build_store(profile, persist_directory=target)
        return store

    def switch_index(self, name: str):
        """Set ``name`` active and return ``(profile, store)``."""
        profile = self.registry.set_active(name)
        store = self.load_index(name)
        logger.info("Switched active index to %r", name)
        return profile, store

    def delete_index(self, name: str) -> IndexProfile:
        """Remove the profile from the registry and delete its directory."""
        profile = self.registry.remove(name)
        target = self.index_dir(name)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        logger.info("Deleted index %r", name)
        return profile

    # ------------------------------------------------------------------ #
    # Export / import
    # ------------------------------------------------------------------ #

    def export_index(self, name: str, dest: Path) -> Path:
        """Archive an index (data + profile) to a ``.zip`` at ``dest``.

        ``dest`` may be a directory (the archive is named ``<name>.zip``) or a
        full ``.zip`` path. Returns the archive path.
        """
        profile = self.registry.get(name)
        src_dir = self.index_dir(name)
        if not src_dir.exists():
            raise IndexRegistryError(f"Index directory missing for {name!r}")

        dest = Path(dest)
        if dest.is_dir() or dest.suffix != ".zip":
            dest.mkdir(parents=True, exist_ok=True)
            archive_path = dest / f"{name}.zip"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            archive_path = dest

        import json

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Embed the profile so an import is fully self-describing.
            zf.writestr(PROFILE_ARCHIVE_NAME, json.dumps(profile.to_dict(), indent=2))
            for file in src_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=str(Path("data") / file.relative_to(src_dir)))
        logger.info("Exported index %r to %s", name, archive_path)
        return archive_path

    def import_index(
        self,
        src: Path,
        name: Optional[str] = None,
        *,
        make_active: bool = False,
        overwrite: bool = False,
    ) -> IndexProfile:
        """Restore an archived index. ``name`` overrides the archived name."""
        src = Path(src)
        if not src.exists():
            raise IndexRegistryError(f"Archive not found: {src}")

        import json

        with zipfile.ZipFile(src, "r") as zf:
            # Guard against zip-slip: no member may escape the extraction root.
            for member in zf.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise IndexRegistryError(f"Unsafe path in archive: {member!r}")
            try:
                profile_data = json.loads(zf.read(PROFILE_ARCHIVE_NAME).decode("utf-8"))
            except KeyError as exc:
                raise IndexRegistryError(
                    f"Archive missing {PROFILE_ARCHIVE_NAME}; not a VectorFlow index export"
                ) from exc

            profile = IndexProfile.from_dict(profile_data)
            if name is not None:
                profile.name = name
            validate_index_name(profile.name)

            if self.registry.exists(profile.name) and not overwrite:
                raise IndexRegistryError(f"Index {profile.name!r} already exists")

            target = self.index_dir(profile.name)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp)
                data_dir = Path(tmp) / "data"
                if data_dir.exists():
                    for file in data_dir.rglob("*"):
                        if file.is_file():
                            rel = file.relative_to(data_dir)
                            out = target / rel
                            out.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file, out)

        profile.touch()
        self.registry.register(profile, make_active=make_active, overwrite=overwrite)
        logger.info("Imported index %r from %s", profile.name, src)
        return profile

    # ------------------------------------------------------------------ #
    # Train / benchmark (extended by later phases)
    # ------------------------------------------------------------------ #

    def train_index(self, name: str, training_vectors: Optional[np.ndarray] = None) -> IndexProfile:
        """Train a trainable index recipe (IVF/PQ families).

        Phase 12e: HNSW/Flat indexes need no training, so this is a no-op for
        them and returns the profile unchanged. Phase 12f wires real training
        for IVF/PQ recipes that require it.
        """
        profile = self.registry.get(name)
        logger.debug("train_index(%r): backend=%s type=%s (no-op in 12e for this type)",
                     name, profile.backend, profile.index_type)
        return profile

    def benchmark_index(self, name: str, *args, **kwargs):
        """Benchmark an index — implemented in Phase 12i."""
        raise NotImplementedError("Index benchmarking lands in Phase 12i")

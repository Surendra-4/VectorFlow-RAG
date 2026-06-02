# src/indexing/registry.py

"""
IndexRegistry (Phase 12e) — the catalog of named :class:`IndexProfile` entities.

Persisted to ``<project_root>/var/index_registry.json``. Tracks:

* every registered index by name,
* which one is currently *active* (the index retrieval reads from).

Thread-safe (RLock). The registry stores only metadata — the actual vector
data lives under ``indices/<name>/`` and is owned by the IndexManager.

Index names are validated to a filesystem-safe charset so a name always maps
1:1 to a directory without escaping the indices root.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from src.indexing.profile import IndexProfile
from src.logging_setup import get_logger

logger = get_logger(__name__)

REGISTRY_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class IndexRegistryError(ValueError):
    """Raised for invalid names, duplicates, or missing indexes."""


def validate_index_name(name: str) -> str:
    """Return ``name`` if it's a safe index id, else raise.

    Allowed: starts alphanumeric; up to 64 chars of ``[A-Za-z0-9._-]``. This
    bans path separators and ``..`` so a name can never escape the indices
    root directory.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise IndexRegistryError(
            f"Invalid index name {name!r}: must match {_NAME_RE.pattern}"
        )
    if name in (".", ".."):  # defensive — already excluded by the regex
        raise IndexRegistryError(f"Invalid index name {name!r}")
    return name


class IndexRegistry:
    """Thread-safe catalog of named index profiles + active pointer."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            from src.config import get_settings

            path = Path(get_settings().app.project_root) / "var" / "index_registry.json"
        self.path = Path(path)
        self._lock = threading.RLock()
        self._profiles: Dict[str, IndexProfile] = {}
        self._active: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def list_profiles(self) -> List[IndexProfile]:
        with self._lock:
            return list(self._profiles.values())

    def names(self) -> List[str]:
        with self._lock:
            return list(self._profiles.keys())

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._profiles

    def get(self, name: str) -> IndexProfile:
        with self._lock:
            if name not in self._profiles:
                raise IndexRegistryError(f"No such index: {name!r}")
            return self._profiles[name]

    @property
    def active_name(self) -> Optional[str]:
        with self._lock:
            return self._active

    def get_active(self) -> Optional[IndexProfile]:
        with self._lock:
            if self._active is None:
                return None
            return self._profiles.get(self._active)

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    def register(self, profile: IndexProfile, *, make_active: bool = False,
                 overwrite: bool = False) -> IndexProfile:
        validate_index_name(profile.name)
        with self._lock:
            if profile.name in self._profiles and not overwrite:
                raise IndexRegistryError(f"Index {profile.name!r} already exists")
            self._profiles[profile.name] = profile
            if make_active or self._active is None:
                self._active = profile.name
            self._persist()
            return profile

    def update(self, profile: IndexProfile) -> IndexProfile:
        with self._lock:
            if profile.name not in self._profiles:
                raise IndexRegistryError(f"No such index: {profile.name!r}")
            self._profiles[profile.name] = profile
            self._persist()
            return profile

    def set_active(self, name: str) -> IndexProfile:
        with self._lock:
            if name not in self._profiles:
                raise IndexRegistryError(f"No such index: {name!r}")
            self._active = name
            self._profiles[name].touch()
            self._persist()
            return self._profiles[name]

    def remove(self, name: str) -> IndexProfile:
        with self._lock:
            if name not in self._profiles:
                raise IndexRegistryError(f"No such index: {name!r}")
            profile = self._profiles.pop(name)
            if self._active == name:
                # Pick any remaining index as active, else None.
                self._active = next(iter(self._profiles), None)
            self._persist()
            return profile

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REGISTRY_VERSION,
            "active": self._active,
            "profiles": {n: p.to_dict() for n, p in self._profiles.items()},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read index registry (starting empty): %s", exc)
            return
        if data.get("version") != REGISTRY_VERSION:
            logger.warning("Index registry version mismatch; ignoring file.")
            return
        profiles = data.get("profiles", {})
        for name, pd in profiles.items():
            try:
                self._profiles[name] = IndexProfile.from_dict(pd)
            except Exception as exc:
                logger.warning("Skipping malformed index profile %r: %s", name, exc)
        active = data.get("active")
        if active in self._profiles:
            self._active = active
        elif self._profiles:
            self._active = next(iter(self._profiles))

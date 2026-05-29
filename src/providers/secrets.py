# src/providers/secrets.py

"""
Backend-only secret storage for online-provider API keys (Phase 12).

Security contract (enforced by tests):

* Keys are persisted **server-side only** — never returned to the frontend,
  never written to ``localStorage``. The API exposes only ``configured: bool``
  plus a redacted hint (last 4 chars).
* At-rest encryption via ``cryptography.Fernet`` when available. The key comes
  from ``$VFR_SECRET_KEY`` (a urlsafe-base64 Fernet key) or, failing that, an
  auto-generated keyfile (``0600``) beside the store. If ``cryptography`` is
  not installed we degrade to an obfuscated-but-unencrypted file (``0600``)
  and log a one-time warning — the platform still works locally.
* Values are never logged. :func:`redact_secret` is the only thing that turns
  a key into a printable string, and it always masks.

File format (JSON)::

    {"version": 1, "encrypted": true, "secrets": {"openai": "<token>", ...}}

where ``<token>`` is Fernet ciphertext (encrypted) or base64 (fallback).
"""

from __future__ import annotations

import base64
import json
import os
import stat
import threading
from pathlib import Path
from typing import Dict, Optional

from src.logging_setup import get_logger

logger = get_logger(__name__)

SECRETS_VERSION = 1
_OBFUSCATION_WARNED = False


def redact_secret(value: Optional[str]) -> str:
    """Turn a secret into a safe, printable hint. Always masks the body."""
    if not value:
        return "<unset>"
    if len(value) <= 4:
        return "****"
    return f"****{value[-4:]}"


def _restrict_perms(path: Path) -> None:
    """Best-effort chmod to owner-only (0600). No-op on platforms w/o chmod."""
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pass


class SecretStore:
    """Thread-safe, file-backed store for provider API keys.

    Parameters
    ----------
    path:
        JSON file to persist to. Defaults to ``<project_root>/var/secrets.json``.
    key_path:
        Where the Fernet keyfile lives when ``$VFR_SECRET_KEY`` is unset.
        Defaults to ``<path>.key``.
    """

    def __init__(self, path: Optional[Path] = None, key_path: Optional[Path] = None):
        if path is None:
            from src.config import get_settings

            path = Path(get_settings().app.project_root) / "var" / "secrets.json"
        self.path = Path(path)
        self.key_path = Path(key_path) if key_path else self.path.with_suffix(".key")
        self._lock = threading.RLock()
        self._fernet = self._init_fernet()
        # In-memory mirror of the on-disk secrets map (plaintext values).
        self._cache: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Crypto setup
    # ------------------------------------------------------------------ #

    @property
    def encrypted(self) -> bool:
        return self._fernet is not None

    def _init_fernet(self):
        """Return a Fernet instance or ``None`` if encryption is unavailable."""
        try:
            from cryptography.fernet import Fernet
        except Exception:
            global _OBFUSCATION_WARNED
            if not _OBFUSCATION_WARNED:
                logger.warning(
                    "cryptography not installed — API keys will be stored "
                    "obfuscated but NOT encrypted. Install `cryptography` for "
                    "at-rest encryption."
                )
                _OBFUSCATION_WARNED = True
            return None

        key = self._resolve_fernet_key(Fernet)
        return Fernet(key)

    def _resolve_fernet_key(self, Fernet) -> bytes:
        """Resolve a Fernet key from env, else a persisted keyfile, else new."""
        env_key = os.environ.get("VFR_SECRET_KEY")
        if env_key:
            return env_key.encode("utf-8")

        if self.key_path.exists():
            return self.key_path.read_bytes().strip()

        # Generate and persist a fresh key with restrictive perms.
        key = Fernet.generate_key()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_bytes(key)
        _restrict_perms(self.key_path)
        logger.info("Generated new secret-store encryption key at %s", self.key_path)
        return key

    # ------------------------------------------------------------------ #
    # Encode/decode a single value
    # ------------------------------------------------------------------ #

    def _encode(self, plaintext: str) -> str:
        raw = plaintext.encode("utf-8")
        if self._fernet is not None:
            return self._fernet.encrypt(raw).decode("ascii")
        # Fallback: base64 obfuscation (NOT security; just not plaintext-on-disk).
        return base64.b64encode(raw).decode("ascii")

    def _decode(self, token: str) -> Optional[str]:
        try:
            if self._fernet is not None:
                return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
            return base64.b64decode(token.encode("ascii")).decode("utf-8")
        except Exception as exc:
            # A key rotation or corrupted entry — drop it rather than crash.
            logger.warning("Could not decode a stored secret (skipping): %s", type(exc).__name__)
            return None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        with self._lock:
            self._cache = {}
            if not self.path.exists():
                return
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to read secret store, starting empty: %s", exc)
                return
            stored = data.get("secrets", {})
            for provider, token in stored.items():
                value = self._decode(token)
                if value is not None:
                    self._cache[provider] = value

    def _persist(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": SECRETS_VERSION,
                "encrypted": self.encrypted,
                "secrets": {p: self._encode(v) for p, v in self._cache.items()},
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            _restrict_perms(tmp)
            os.replace(tmp, self.path)
            _restrict_perms(self.path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_secret(self, provider: str, value: str) -> None:
        if not provider:
            raise ValueError("provider name is required")
        if not value:
            raise ValueError("secret value must be non-empty")
        with self._lock:
            self._cache[provider] = value
            self._persist()
        logger.info("Stored API key for provider=%s (%s)", provider, redact_secret(value))

    def get_secret(self, provider: str) -> Optional[str]:
        with self._lock:
            return self._cache.get(provider)

    def has_secret(self, provider: str) -> bool:
        with self._lock:
            return provider in self._cache

    def delete_secret(self, provider: str) -> bool:
        with self._lock:
            existed = provider in self._cache
            self._cache.pop(provider, None)
            if existed:
                self._persist()
        if existed:
            logger.info("Deleted API key for provider=%s", provider)
        return existed

    def describe(self) -> Dict[str, Dict[str, object]]:
        """Frontend-safe view: which providers are configured + a redacted hint.

        NEVER returns the raw key. This is the only thing the API surfaces.
        """
        with self._lock:
            return {
                provider: {"configured": True, "hint": redact_secret(value)}
                for provider, value in self._cache.items()
            }


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #

_STORE: Optional[SecretStore] = None
_STORE_LOCK = threading.Lock()


def get_secret_store() -> SecretStore:
    """Return the process-wide SecretStore singleton."""
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = SecretStore()
    return _STORE


def reset_secret_store() -> None:
    """Drop the singleton — primarily for tests that use a temp path."""
    global _STORE
    with _STORE_LOCK:
        _STORE = None

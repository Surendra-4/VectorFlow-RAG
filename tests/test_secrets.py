# tests/test_secrets.py

"""Tests for the backend-only SecretStore (Phase 12a)."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from src.providers.secrets import SecretStore, redact_secret


@pytest.fixture
def store(temp_dir):
    return SecretStore(path=temp_dir / "secrets.json", key_path=temp_dir / "secret.key")


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_redact_masks_body():
    assert redact_secret(None) == "<unset>"
    assert redact_secret("") == "<unset>"
    assert redact_secret("abcd") == "****"
    assert redact_secret("sk-1234567890") == "****7890"


# --------------------------------------------------------------------------- #
# CRUD roundtrip
# --------------------------------------------------------------------------- #


def test_set_get_has_delete(store):
    assert store.has_secret("openai") is False
    assert store.get_secret("openai") is None

    store.set_secret("openai", "sk-abc12345")
    assert store.has_secret("openai") is True
    assert store.get_secret("openai") == "sk-abc12345"

    assert store.delete_secret("openai") is True
    assert store.has_secret("openai") is False
    assert store.delete_secret("openai") is False  # already gone


def test_set_rejects_empty(store):
    with pytest.raises(ValueError):
        store.set_secret("openai", "")
    with pytest.raises(ValueError):
        store.set_secret("", "x")


def test_describe_never_leaks_raw_value(store):
    store.set_secret("anthropic", "sk-ant-supersecret")
    desc = store.describe()
    assert desc["anthropic"]["configured"] is True
    assert desc["anthropic"]["hint"] == "****cret"
    # The raw value must not appear anywhere in the describe() output.
    assert "supersecret" not in json.dumps(desc)


# --------------------------------------------------------------------------- #
# Persistence + at-rest protection
# --------------------------------------------------------------------------- #


def test_persistence_across_instances(temp_dir):
    p = temp_dir / "secrets.json"
    k = temp_dir / "secret.key"
    s1 = SecretStore(path=p, key_path=k)
    s1.set_secret("groq", "gsk-persist-me")

    s2 = SecretStore(path=p, key_path=k)
    assert s2.get_secret("groq") == "gsk-persist-me"


def test_value_not_plaintext_on_disk(store):
    store.set_secret("openai", "sk-PLAINTEXTLEAK")
    raw = store.path.read_text(encoding="utf-8")
    assert "sk-PLAINTEXTLEAK" not in raw
    # cryptography is installed in this env → store must be encrypted.
    assert store.encrypted is True
    payload = json.loads(raw)
    assert payload["encrypted"] is True
    assert payload["version"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
def test_file_perms_are_owner_only(store):
    store.set_secret("openai", "sk-perms")
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600


def test_env_key_used_when_present(temp_dir, monkeypatch):
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("VFR_SECRET_KEY", key)
    # No keyfile should be created when env key is supplied.
    p = temp_dir / "secrets.json"
    kf = temp_dir / "secret.key"
    s = SecretStore(path=p, key_path=kf)
    s.set_secret("openai", "sk-env")
    assert s.encrypted is True
    assert not kf.exists()
    # A second store reading the same env key decrypts successfully.
    s2 = SecretStore(path=p, key_path=kf)
    assert s2.get_secret("openai") == "sk-env"


def test_corrupt_entry_is_dropped_not_fatal(temp_dir):
    p = temp_dir / "secrets.json"
    p.write_text(json.dumps({
        "version": 1, "encrypted": True,
        "secrets": {"openai": "not-valid-fernet-token"},
    }), encoding="utf-8")
    # Should load without raising; the bad entry is simply absent.
    s = SecretStore(path=p, key_path=temp_dir / "secret.key")
    assert s.has_secret("openai") is False

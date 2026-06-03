# tests/test_auth_api.py

"""Email/password auth API tests (Phase 14b). Temp SQLite; no network."""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.db.session import reset_engine


@pytest.fixture
def client(temp_dir, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{temp_dir}/auth.db")
    reset_engine()
    app = create_app(init_pipeline=False)
    with TestClient(app) as c:
        yield c
    reset_engine()


def _signup(client, email="user@example.com", password="hunter2pass", name="User"):
    return client.post("/api/v1/auth/signup", json={"email": email, "password": password, "name": name})


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


def test_providers_password_only_by_default(client):
    r = client.get("/api/v1/auth/providers")
    assert r.status_code == 200
    body = r.json()
    assert body["password"] is True
    assert body["google"] is False and body["github"] is False
    assert body["auth_required"] is False


# --------------------------------------------------------------------------- #
# Signup / login
# --------------------------------------------------------------------------- #


def test_signup_returns_token_and_user(client):
    r = _signup(client)
    assert r.status_code == 201
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["email"] == "user@example.com"
    assert "password_hash" not in body["user"]


def test_signup_duplicate_email_409(client):
    _signup(client)
    r = _signup(client)
    assert r.status_code == 409
    assert r.json()["code"] == "email_taken"


def test_signup_short_password_422(client):
    r = client.post("/api/v1/auth/signup", json={"email": "x@example.com", "password": "short"})
    assert r.status_code == 422


def test_login_success_and_failure(client):
    _signup(client, email="a@example.com", password="correct-horse")
    ok = client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "correct-horse"})
    assert ok.status_code == 200
    assert ok.json()["access_token"]

    bad = client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "wrong"})
    assert bad.status_code == 401
    assert bad.json()["code"] == "invalid_credentials"

    missing = client.post("/api/v1/auth/login", json={"email": "nope@example.com", "password": "whatever"})
    assert missing.status_code == 401


# --------------------------------------------------------------------------- #
# me / auth guard
# --------------------------------------------------------------------------- #


def test_me_requires_token(client):
    assert client.get("/api/v1/auth/me").status_code == 401


def test_me_with_token(client):
    token = _signup(client).json()["access_token"]
    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["user"]["email"] == "user@example.com"


def test_me_rejects_garbage_token(client):
    r = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


# --------------------------------------------------------------------------- #
# Password reset
# --------------------------------------------------------------------------- #


def test_reset_flow_end_to_end(client):
    _signup(client, email="r@example.com", password="old-password-1")
    # request → opaque message + dev reset link (no SMTP configured)
    req = client.post("/api/v1/auth/reset/request", json={"email": "r@example.com"})
    assert req.status_code == 200
    link = req.json()["reset_link"]
    assert link and "token=" in link
    token = link.split("token=")[1]

    # confirm with new password → returns a fresh token
    conf = client.post("/api/v1/auth/reset/confirm", json={"token": token, "password": "new-password-2"})
    assert conf.status_code == 200
    assert conf.json()["access_token"]

    # old password no longer works; new one does
    assert client.post("/api/v1/auth/login", json={"email": "r@example.com", "password": "old-password-1"}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"email": "r@example.com", "password": "new-password-2"}).status_code == 200


def test_reset_request_unknown_email_is_opaque(client):
    r = client.post("/api/v1/auth/reset/request", json={"email": "ghost@example.com"})
    assert r.status_code == 200
    assert r.json()["reset_link"] is None  # no account → no link, but still 200


def test_reset_confirm_bad_token(client):
    r = client.post("/api/v1/auth/reset/confirm", json={"token": "invalidtoken", "password": "whatever12"})
    assert r.status_code == 400
    assert r.json()["code"] in ("invalid_reset", "expired_reset")


# --------------------------------------------------------------------------- #
# Per-user stats
# --------------------------------------------------------------------------- #


def test_stats_get_and_reset(client):
    token = _signup(client).json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}
    r = client.get("/api/v1/auth/me/stats", headers=h)
    assert r.status_code == 200
    assert r.json()["stats"]["searches"] == 0

    rr = client.post("/api/v1/auth/me/stats/reset", headers=h)
    assert rr.status_code == 200
    assert rr.json()["stats"]["reset_at"] is not None


def test_stats_requires_auth(client):
    assert client.get("/api/v1/auth/me/stats").status_code == 401


# --------------------------------------------------------------------------- #
# OpenAPI
# --------------------------------------------------------------------------- #


def test_auth_routes_in_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/api/v1/auth/signup", "/api/v1/auth/login", "/api/v1/auth/me",
              "/api/v1/auth/reset/request", "/api/v1/auth/providers"]:
        assert p in paths

# tests/test_auth_oauth.py

"""OAuth (Google/GitHub) flow tests (Phase 14c). No real provider calls."""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import src.auth.oauth as oauth_mod
from src.api.app import create_app
from src.auth.oauth import OAuthUser
from src.config import reset_settings_cache
from src.db.session import reset_engine


@pytest.fixture
def client(temp_dir, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{temp_dir}/oauth.db")
    # Configure both providers + a known public/frontend URL.
    monkeypatch.setenv("VFR_AUTH__GOOGLE_CLIENT_ID", "gid")
    monkeypatch.setenv("VFR_AUTH__GOOGLE_CLIENT_SECRET", "gsecret")
    monkeypatch.setenv("VFR_AUTH__GITHUB_CLIENT_ID", "hid")
    monkeypatch.setenv("VFR_AUTH__GITHUB_CLIENT_SECRET", "hsecret")
    monkeypatch.setenv("VFR_AUTH__PUBLIC_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("VFR_AUTH__FRONTEND_URL", "https://app.example.com")
    reset_settings_cache()
    reset_engine()
    app = create_app(init_pipeline=False)
    # Don't auto-follow the external provider redirect.
    with TestClient(app, follow_redirects=False) as c:
        yield c
    reset_settings_cache()
    reset_engine()


# --------------------------------------------------------------------------- #
# authorize_url builder (unit)
# --------------------------------------------------------------------------- #


def test_authorize_url_contains_required_params(client):
    url = oauth_mod.authorize_url("google", "state123")
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=gid" in url
    assert "state=state123" in url
    assert "redirect_uri=https%3A%2F%2Fapi.example.com%2Fapi%2Fv1%2Fauth%2Fgoogle%2Fcallback" in url


def test_redirect_uri(client):
    assert oauth_mod.redirect_uri_for("github") == "https://api.example.com/api/v1/auth/github/callback"


# --------------------------------------------------------------------------- #
# providers + start
# --------------------------------------------------------------------------- #


def test_providers_reflect_configuration(client):
    body = client.get("/api/v1/auth/providers").json()
    assert body["google"] is True and body["github"] is True


def test_oauth_start_redirects_with_state_cookie(client):
    r = client.get("/api/v1/auth/google")
    assert r.status_code == 307
    assert r.headers["location"].startswith("https://accounts.google.com/")
    assert "vfr_oauth_state" in r.headers.get("set-cookie", "")


def test_oauth_start_unknown_provider_404(client):
    assert client.get("/api/v1/auth/notaprovider").status_code == 404


def test_oauth_start_unconfigured_503(client, monkeypatch):
    monkeypatch.delenv("VFR_AUTH__GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("VFR_AUTH__GOOGLE_CLIENT_SECRET", raising=False)
    reset_settings_cache()
    assert client.get("/api/v1/auth/google").status_code == 503


# --------------------------------------------------------------------------- #
# callback
# --------------------------------------------------------------------------- #


def test_callback_happy_path_creates_user_and_redirects(client, monkeypatch):
    monkeypatch.setattr(oauth_mod, "exchange_code", lambda provider, code: "tok")
    monkeypatch.setattr(
        oauth_mod, "fetch_user",
        lambda provider, token: OAuthUser(email="oauth@example.com", name="OAuth User", avatar_url="http://a/x.png"),
    )
    r = client.get(
        "/api/v1/auth/github/callback",
        params={"code": "c", "state": "s1"},
        cookies={"vfr_oauth_state": "s1"},
    )
    assert r.status_code == 307
    loc = r.headers["location"]
    assert loc.startswith("https://app.example.com/auth/callback#access_token=")

    # The user now exists and can be fetched with the issued token.
    token = loc.split("#access_token=")[1]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "oauth@example.com"
    assert me.json()["user"]["provider"] == "github"


def test_callback_state_mismatch_redirects_to_login_error(client):
    r = client.get(
        "/api/v1/auth/google/callback",
        params={"code": "c", "state": "bad"},
        cookies={"vfr_oauth_state": "good"},
    )
    assert r.status_code == 307
    assert "error=state_mismatch" in r.headers["location"]


def test_callback_provider_error_redirects(client):
    r = client.get("/api/v1/auth/google/callback", params={"error": "access_denied"})
    assert r.status_code == 307
    assert "error=access_denied" in r.headers["location"]


def test_callback_oauth_failure_redirects(client, monkeypatch):
    def boom(provider, code):
        raise oauth_mod.OAuthError("token bad", provider=provider)

    monkeypatch.setattr(oauth_mod, "exchange_code", boom)
    r = client.get(
        "/api/v1/auth/google/callback",
        params={"code": "c", "state": "s"},
        cookies={"vfr_oauth_state": "s"},
    )
    assert r.status_code == 307
    assert "error=oauth_error" in r.headers["location"]

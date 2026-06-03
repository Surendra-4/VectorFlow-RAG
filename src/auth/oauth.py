# src/auth/oauth.py

"""
Google + GitHub sign-in via the OAuth 2.0 authorization-code flow (Phase 14c).

Implemented directly over ``requests`` — no Authlib/SDK — so the dependency
surface stays flat and the failure modes are uniform. Each provider is just a
set of endpoints + a userinfo normalizer. Credentials come from settings
(operator-provided); an unconfigured provider is simply unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


class OAuthError(Exception):
    def __init__(self, message: str, *, provider: str = ""):
        super().__init__(message)
        self.message = message
        self.provider = provider


@dataclass
class OAuthUser:
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None


_PROVIDERS = {
    "google": {
        "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
        "token": "https://oauth2.googleapis.com/token",
        "userinfo": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
    },
    "github": {
        "authorize": "https://github.com/login/oauth/authorize",
        "token": "https://github.com/login/oauth/access_token",
        "userinfo": "https://api.github.com/user",
        "emails": "https://api.github.com/user/emails",
        "scope": "read:user user:email",
    },
}


def supported(provider: str) -> bool:
    return provider in _PROVIDERS


def _creds(provider: str) -> tuple[Optional[str], Optional[str]]:
    cfg = get_settings().auth
    if provider == "google":
        return cfg.google_client_id, cfg.google_client_secret
    if provider == "github":
        return cfg.github_client_id, cfg.github_client_secret
    return None, None


def is_configured(provider: str) -> bool:
    cid, secret = _creds(provider)
    return bool(cid and secret)


def redirect_uri_for(provider: str) -> str:
    base = get_settings().auth.public_base_url.rstrip("/")
    return f"{base}/api/v1/auth/{provider}/callback"


def authorize_url(provider: str, state: str) -> str:
    if not supported(provider):
        raise OAuthError(f"Unknown provider {provider!r}", provider=provider)
    if not is_configured(provider):
        raise OAuthError(f"{provider} sign-in is not configured.", provider=provider)
    from urllib.parse import urlencode

    cid, _ = _creds(provider)
    p = _PROVIDERS[provider]
    params = {
        "client_id": cid,
        "redirect_uri": redirect_uri_for(provider),
        "scope": p["scope"],
        "state": state,
        "response_type": "code",
    }
    if provider == "google":
        params["access_type"] = "offline"
        params["prompt"] = "select_account"
    return f"{p['authorize']}?{urlencode(params)}"


def exchange_code(provider: str, code: str) -> str:
    """Exchange an authorization code for an access token."""
    cid, secret = _creds(provider)
    p = _PROVIDERS[provider]
    data = {
        "client_id": cid,
        "client_secret": secret,
        "code": code,
        "redirect_uri": redirect_uri_for(provider),
        "grant_type": "authorization_code",
    }
    try:
        resp = requests.post(p["token"], data=data, headers={"Accept": "application/json"}, timeout=15)
        resp.raise_for_status()
        token = resp.json().get("access_token")
    except Exception as exc:
        raise OAuthError(f"Token exchange failed: {exc}", provider=provider) from exc
    if not token:
        raise OAuthError("No access token returned by provider.", provider=provider)
    return token


def fetch_user(provider: str, access_token: str) -> OAuthUser:
    p = _PROVIDERS[provider]
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    try:
        resp = requests.get(p["userinfo"], headers=headers, timeout=15)
        resp.raise_for_status()
        info = resp.json()
    except Exception as exc:
        raise OAuthError(f"Failed to fetch profile: {exc}", provider=provider) from exc

    if provider == "google":
        email = info.get("email")
        if not email:
            raise OAuthError("Google account has no email.", provider=provider)
        return OAuthUser(email=email, name=info.get("name"), avatar_url=info.get("picture"))

    # GitHub: primary email may be private → fetch the emails endpoint.
    email = info.get("email")
    if not email:
        try:
            er = requests.get(p["emails"], headers=headers, timeout=15)
            er.raise_for_status()
            emails = er.json()
            primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
            email = (primary or (emails[0] if emails else {})).get("email")
        except Exception as exc:  # pragma: no cover - network dependent
            raise OAuthError(f"Failed to fetch GitHub email: {exc}", provider=provider) from exc
    if not email:
        raise OAuthError("No verified email on the GitHub account.", provider=provider)
    return OAuthUser(
        email=email,
        name=info.get("name") or info.get("login"),
        avatar_url=info.get("avatar_url"),
    )

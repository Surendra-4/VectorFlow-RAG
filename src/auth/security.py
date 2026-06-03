# src/auth/security.py

"""
Password hashing (bcrypt) + JWT issue/verify (Phase 14).

Security notes:

* bcrypt silently ignores bytes past 72, so we truncate the UTF-8 encoding to
  72 bytes consistently in both hash and verify — no surprise mismatches.
* JWTs are HS256-signed with ``settings.auth.jwt_secret``. Production MUST set a
  long random secret (the default is intentionally obvious so it fails review).
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

from src.config import get_settings


def _clip(password: str) -> bytes:
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_clip(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: Optional[str]) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(_clip(password), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, *, extra: Optional[dict] = None) -> str:
    cfg = get_settings().auth
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=cfg.jwt_expiry_minutes),
        "type": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def decode_token(token: str) -> Optional[dict]:
    cfg = get_settings().auth
    try:
        return jwt.decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm])
    except jwt.PyJWTError:
        return None


def generate_reset_token() -> str:
    """URL-safe, single-use password-reset token."""
    return secrets.token_urlsafe(32)


def generate_state() -> str:
    """CSRF state value for the OAuth authorization-code flow."""
    return secrets.token_urlsafe(24)

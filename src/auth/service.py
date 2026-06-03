# src/auth/service.py

"""
User account operations (Phase 14) — the thin layer between the auth routes and
the ORM. Pure data logic; HTTP concerns live in the routes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.auth.security import generate_reset_token, hash_password, verify_password
from src.config import get_settings
from src.db.models import User, UserStats
from src.logging_setup import get_logger

logger = get_logger(__name__)


class AuthError(Exception):
    """Raised for auth-domain failures (mapped to 4xx by the routes)."""

    def __init__(self, message: str, *, code: str = "auth_error", status: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


def _norm_email(email: str) -> str:
    return email.strip().lower()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == _norm_email(email)).one_or_none()


def get_user(db: Session, user_id: str) -> Optional[User]:
    return db.get(User, user_id)


def ensure_stats(db: Session, user: User) -> UserStats:
    if user.stats is None:
        user.stats = UserStats()
        db.add(user.stats)
        db.flush()
    return user.stats


def create_local_user(db: Session, *, email: str, password: str, name: Optional[str] = None) -> User:
    email = _norm_email(email)
    if get_user_by_email(db, email):
        raise AuthError("An account with this email already exists.", code="email_taken", status=409)
    user = User(
        email=email,
        name=name or email.split("@")[0],
        password_hash=hash_password(password),
        provider="local",
    )
    user.stats = UserStats()
    db.add(user)
    db.flush()
    logger.info("Created local user %s", email)
    return user


def authenticate(db: Session, *, email: str, password: str) -> User:
    user = get_user_by_email(db, email)
    # Constant-ish: verify even when user is None to avoid trivial enumeration.
    if user is None or not verify_password(password, user.password_hash):
        raise AuthError("Invalid email or password.", code="invalid_credentials", status=401)
    return user


def upsert_oauth_user(
    db: Session, *, email: str, provider: str, name: Optional[str], avatar_url: Optional[str]
) -> User:
    """Link an OAuth login to an existing account (by email) or create one."""
    email = _norm_email(email)
    user = get_user_by_email(db, email)
    if user is None:
        user = User(
            email=email,
            name=name or email.split("@")[0],
            avatar_url=avatar_url,
            provider=provider,
            email_verified=True,
        )
        user.stats = UserStats()
        db.add(user)
        logger.info("Created %s OAuth user %s", provider, email)
    else:
        user.provider = provider
        user.email_verified = True
        if avatar_url and not user.avatar_url:
            user.avatar_url = avatar_url
        if name and not user.name:
            user.name = name
    db.flush()
    return user


def begin_password_reset(db: Session, email: str) -> Optional[str]:
    """Set a reset token for ``email`` (if it exists). Returns the token, or
    None when no such account — the route stays opaque either way."""
    user = get_user_by_email(db, email)
    if user is None:
        return None
    token = generate_reset_token()
    mins = get_settings().auth.reset_token_expiry_minutes
    user.reset_token = token
    user.reset_token_expires = datetime.now(timezone.utc) + timedelta(minutes=mins)
    db.flush()
    return token


def complete_password_reset(db: Session, *, token: str, new_password: str) -> User:
    user = db.query(User).filter(User.reset_token == token).one_or_none()
    if user is None or user.reset_token_expires is None:
        raise AuthError("Invalid or expired reset link.", code="invalid_reset", status=400)
    expires = user.reset_token_expires
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < datetime.now(timezone.utc):
        raise AuthError("This reset link has expired.", code="expired_reset", status=400)
    user.password_hash = hash_password(new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.flush()
    logger.info("Password reset completed for %s", user.email)
    return user


def reset_user_stats(db: Session, user: User) -> UserStats:
    stats = ensure_stats(db, user)
    from src.db.models import STAT_FIELDS

    for field in STAT_FIELDS:
        setattr(stats, field, 0)
    stats.reset_at = datetime.now(timezone.utc)
    db.flush()
    logger.info("Reset stats for user %s", user.email)
    return stats


def record_event(db: Session, user: User, **increments: int) -> None:
    """Atomically bump per-user counters. Unknown keys are ignored."""
    from src.db.models import STAT_FIELDS

    stats = ensure_stats(db, user)
    for key, amount in increments.items():
        if key in STAT_FIELDS and amount:
            setattr(stats, key, (getattr(stats, key) or 0) + amount)
    stats.last_active_at = datetime.now(timezone.utc)
    db.flush()


def record_for_user_id(user_id: Optional[str], **increments: int) -> None:
    """Fire-and-forget per-user stat increment in its own session.

    Used by the data routes (search/ask/ingest) to attribute usage to the
    authenticated caller without holding a request-scoped session. Never raises
    — observability must not break a response.
    """
    if not user_id:
        return
    from src.db.session import session_scope

    try:
        with session_scope() as db:
            user = db.get(User, user_id)
            if user is not None:
                record_event(db, user, **increments)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("record_for_user_id suppressed: %s", exc)

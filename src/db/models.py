# src/db/models.py

"""
ORM models for accounts + per-user statistics (Phase 14).

Scope, deliberately small: this database stores **only** user identities and
their usage statistics. Ingested documents, chunks, embeddings, and indexes
are NEVER persisted here — they stay in the vector store / local index files.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Null for OAuth-only accounts; a bcrypt hash otherwise.
    password_hash: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # How the account was created / last authenticated: local | google | github.
    provider: Mapped[str] = mapped_column(String(20), default="local", nullable=False)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Password reset (single active token).
    reset_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    reset_token_expires: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    stats: Mapped["UserStats"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )

    def to_public(self) -> dict:
        """Safe, client-facing view — never includes the password hash."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "provider": self.provider,
            "email_verified": self.email_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Counter columns that the stats API exposes + the reset zeroes.
STAT_FIELDS = (
    "searches",
    "asks",
    "retrievals",
    "documents_ingested",
    "chunks_ingested",
    "cache_hits",
    "tokens_generated",
)


class UserStats(Base):
    __tablename__ = "user_stats"

    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    searches: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    asks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retrievals: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    documents_ingested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks_ingested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_generated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    reset_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="stats")

    def to_dict(self) -> dict:
        return {
            **{f: getattr(self, f) for f in STAT_FIELDS},
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "reset_at": self.reset_at.isoformat() if self.reset_at else None,
        }

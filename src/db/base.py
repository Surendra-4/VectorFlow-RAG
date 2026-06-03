# src/db/base.py

"""Declarative base for all ORM models (Phase 14)."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared SQLAlchemy 2.0 declarative base."""

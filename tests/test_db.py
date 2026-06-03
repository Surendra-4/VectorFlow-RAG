# tests/test_db.py

"""DB layer smoke tests (Phase 14a) — temp SQLite, no Postgres needed."""

from __future__ import annotations

import os

import pytest

from src.db import User, UserStats, reset_engine, resolve_database_url, session_scope
from src.db.session import init_db


@pytest.fixture
def temp_db(temp_dir, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{temp_dir}/test.db")
    reset_engine()
    init_db()
    yield
    reset_engine()


def test_url_normalization(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host:5432/db")
    assert resolve_database_url() == "postgresql+psycopg2://u:p@host:5432/db"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    assert resolve_database_url() == "postgresql+psycopg2://u:p@host/db"


def test_create_user_and_stats(temp_db):
    with session_scope() as db:
        u = User(email="a@example.com", name="Ada", password_hash="x", provider="local")
        u.stats = UserStats()
        db.add(u)

    with session_scope() as db:
        u = db.query(User).filter_by(email="a@example.com").one()
        assert u.id and len(u.id) == 32
        assert u.stats is not None
        assert u.stats.searches == 0
        # public view never leaks the hash
        pub = u.to_public()
        assert "password_hash" not in pub
        assert pub["email"] == "a@example.com"


def test_email_unique(temp_db):
    from sqlalchemy.exc import IntegrityError

    with session_scope() as db:
        db.add(User(email="dup@example.com"))
    with pytest.raises(IntegrityError):
        with session_scope() as db:
            db.add(User(email="dup@example.com"))


def test_stats_cascade_delete(temp_db):
    with session_scope() as db:
        u = User(email="c@example.com")
        u.stats = UserStats(searches=5)
        db.add(u)
    with session_scope() as db:
        u = db.query(User).filter_by(email="c@example.com").one()
        db.delete(u)
    with session_scope() as db:
        assert db.query(UserStats).count() == 0


def test_stats_to_dict(temp_db):
    with session_scope() as db:
        u = User(email="d@example.com")
        u.stats = UserStats(searches=3, asks=2, retrievals=10)
        db.add(u)
    with session_scope() as db:
        s = db.query(UserStats).first()
        d = s.to_dict()
        assert d["searches"] == 3 and d["asks"] == 2 and d["retrievals"] == 10
        assert "last_active_at" in d and "reset_at" in d

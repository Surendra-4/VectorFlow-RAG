# src/auth/__init__.py

"""
Authentication + accounts for VectorFlow-RAG (Phase 14).

Backend-owned: email/password + Google/GitHub OAuth, JWT sessions, password
reset, and per-user statistics — all in the FastAPI service alongside the data
API. Public surface:

* :mod:`src.auth.security` — bcrypt + JWT primitives
* :mod:`src.auth.service` — user operations (AuthError on domain failures)
* :mod:`src.auth.oauth` — Google/GitHub authorization-code flow
"""

from src.auth import service  # noqa: F401
from src.auth.service import AuthError

__all__ = ["AuthError", "service"]

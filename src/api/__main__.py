# src/api/__main__.py

"""
Foolproof launcher: ``python -m src.api``.

The FastAPI app is factory-only (no import-time ``app`` instance, so importing
the module never triggers a model load). That means a bare
``uvicorn src.api.app:app`` fails with ``Attribute "app" not found``. This
entry point wraps the correct factory invocation and pulls host/port/reload
from ``Settings.api`` (overridable via ``VFR_API__*`` env vars), so users
don't have to remember the ``--factory`` flag.

Equivalent explicit command:

    uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import uvicorn

from src.config import get_settings


def main() -> None:
    api = get_settings().api
    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=api.host,
        port=api.port,
        reload=api.reload,
    )


if __name__ == "__main__":
    main()

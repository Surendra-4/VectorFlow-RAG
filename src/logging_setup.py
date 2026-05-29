# src/logging_setup.py

"""
Structured logging for VectorFlow-RAG.

Implements two formatters using only the stdlib (no extra runtime deps):

- ``text``: human-readable, optionally ANSI-colored
- ``json``: line-delimited JSON suitable for log aggregation

Public API:

- :func:`get_logger` — return a module-scoped logger
- :func:`configure_logging` — initialize the root logger explicitly
- :func:`configure_from_settings` — initialize from a ``Settings`` instance
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Internal flag so configuration is idempotent unless explicitly forced.
_CONFIGURED: bool = False

# Attributes that ``LogRecord`` always carries — anything else is a custom
# ``extra=`` payload and should be surfaced in the JSON output.
_RESERVED_LOG_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Line-delimited JSON formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)

        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable formatter with optional ANSI color."""

    DEFAULT_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        super().__init__(self.DEFAULT_FMT, self.DEFAULT_DATEFMT)
        # Only emit color if a TTY is attached, regardless of preference.
        self.use_color = bool(use_color) and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self.use_color:
            return formatted
        color = self.LEVEL_COLORS.get(record.levelname)
        return f"{color}{formatted}{self.RESET}" if color else formatted


def configure_logging(
    level: str = "INFO",
    fmt: str = "text",
    file: Optional[Path] = None,
    use_color: bool = True,
    force: bool = False,
) -> None:
    """
    Configure the root logger.

    Idempotent: subsequent calls are a no-op unless ``force=True``. We do
    NOT short-circuit on ``root.handlers`` alone because frameworks (pytest,
    uvicorn, jupyter) attach their own capture handlers — we still want our
    formatter active alongside them, since adding a handler is harmless and
    side-by-side capture is the conventional pattern.
    """
    global _CONFIGURED

    root = logging.getLogger()

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
    elif _CONFIGURED:
        return

    root.setLevel(level.upper())

    formatter: logging.Formatter = (
        JsonFormatter() if fmt == "json" else TextFormatter(use_color=use_color)
    )

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if file is not None:
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    quiet_noisy_loggers()
    _CONFIGURED = True


# Third-party loggers that emit high-volume INFO chatter unrelated to this
# application's behavior. We raise their threshold to WARNING so genuine
# problems still surface, but startup/operation stays readable.
_NOISY_LOGGERS = {
    "httpx": logging.WARNING,                 # one INFO line per HF Hub HEAD request
    "httpcore": logging.WARNING,
    "jax._src.xla_bridge": logging.ERROR,     # "Unable to initialize backend 'tpu'" probe
    "jax": logging.WARNING,
    "chromadb.telemetry": logging.CRITICAL,   # posthog version-mismatch ERROR spam
    "chromadb.telemetry.product.posthog": logging.CRITICAL,
    "sentence_transformers": logging.WARNING,
    "urllib3": logging.WARNING,
}


def quiet_noisy_loggers() -> None:
    """Raise the level of known-chatty third-party loggers (idempotent)."""
    for name, lvl in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)


def configure_from_settings(settings) -> None:
    """Initialize the root logger from a ``Settings`` instance."""
    cfg = getattr(settings, "logging", None)
    if cfg is None:
        configure_logging()
        return
    configure_logging(
        level=cfg.level,
        fmt=cfg.format,
        file=cfg.file,
        use_color=cfg.use_color,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger that respects the current root configuration."""
    return logging.getLogger(name)

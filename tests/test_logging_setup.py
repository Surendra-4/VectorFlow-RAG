# tests/test_logging_setup.py

"""Tests for the structured logging module."""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import pytest

from src.logging_setup import (
    JsonFormatter,
    TextFormatter,
    configure_from_settings,
    configure_logging,
    get_logger,
)
from src import logging_setup


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Restore root logger between tests so configure_logging is testable."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    for h in list(root.handlers):
        root.removeHandler(h)
    logging_setup._CONFIGURED = False

    yield

    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)
    logging_setup._CONFIGURED = False


def _make_record(level=logging.INFO, message="hello", logger_name="test"):
    return logging.LogRecord(
        name=logger_name, level=level, pathname=__file__, lineno=0,
        msg=message, args=None, exc_info=None,
    )


class TestGetLogger:
    def test_returns_logger_instance(self):
        log = get_logger("vfr.test")
        assert isinstance(log, logging.Logger)
        assert log.name == "vfr.test"

    def test_same_name_returns_same_logger(self):
        a = get_logger("vfr.test.shared")
        b = get_logger("vfr.test.shared")
        assert a is b


class TestConfigureLogging:
    def test_attaches_stream_handler(self):
        configure_logging(level="DEBUG", fmt="text", use_color=False, force=True)
        root = logging.getLogger()
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
        assert root.level == logging.DEBUG

    def test_idempotent_without_force(self):
        configure_logging(level="INFO", force=True)
        n_handlers_first = len(logging.getLogger().handlers)
        configure_logging(level="DEBUG")  # should be a no-op
        assert len(logging.getLogger().handlers) == n_handlers_first

    def test_force_replaces_handlers(self):
        configure_logging(level="INFO", force=True)
        first_handlers = list(logging.getLogger().handlers)
        configure_logging(level="DEBUG", force=True)
        # New handler set, not the same instances.
        assert all(h not in first_handlers for h in logging.getLogger().handlers)

    def test_file_handler_created(self, tmp_path):
        log_file = tmp_path / "nested" / "dir" / "vfr.log"
        configure_logging(level="INFO", file=log_file, force=True)
        get_logger("vfr.test").info("written")
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_file.exists()
        assert "written" in log_file.read_text()


class TestJsonFormatter:
    def test_basic_payload(self):
        rec = _make_record(message="boom")
        out = JsonFormatter().format(rec)
        payload = json.loads(out)
        assert payload["message"] == "boom"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "test"
        assert "ts" in payload

    def test_extra_fields_preserved(self):
        rec = _make_record()
        rec.user_id = 42
        rec.request_id = "abc-123"
        payload = json.loads(JsonFormatter().format(rec))
        assert payload["user_id"] == 42
        assert payload["request_id"] == "abc-123"

    def test_non_serializable_extras_repr_only(self):
        class Thing:
            def __repr__(self): return "<thing>"

        rec = _make_record()
        rec.thing = Thing()
        payload = json.loads(JsonFormatter().format(rec))
        assert payload["thing"] == "<thing>"

    def test_exception_info_included(self):
        try:
            raise ValueError("nope")
        except ValueError:
            rec = logging.LogRecord(
                "test", logging.ERROR, __file__, 0, "failed", None, sys.exc_info(),
            )
        payload = json.loads(JsonFormatter().format(rec))
        assert "ValueError" in payload["exc_info"]


class TestTextFormatter:
    def test_no_color_when_not_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
        fmt = TextFormatter(use_color=True)
        rec = _make_record(message="plain")
        out = fmt.format(rec)
        assert "\033[" not in out
        assert "plain" in out

    def test_includes_level_name_and_logger(self, monkeypatch):
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
        fmt = TextFormatter(use_color=False)
        out = fmt.format(_make_record(level=logging.WARNING, message="hello", logger_name="vfr.x"))
        assert "WARNING" in out
        assert "vfr.x" in out
        assert "hello" in out


class TestConfigureFromSettings:
    def test_defaults_when_no_settings_attribute(self):
        configure_from_settings(object())
        # Just ensure something got attached.
        assert logging.getLogger().handlers

    def test_uses_settings_level(self):
        from src.config import LoggingSettings, Settings

        s = Settings(logging=LoggingSettings(level="ERROR"))
        configure_from_settings(s)
        assert logging.getLogger().level == logging.ERROR

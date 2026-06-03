# src/auth/emails.py

"""
Transactional email (Phase 14) — password-reset delivery.

Uses stdlib ``smtplib`` when SMTP is configured (operator provides host/creds).
With no SMTP configured (local/dev), the reset link is logged instead of sent,
so the flow is fully testable without an email provider.
"""

from __future__ import annotations

from email.mime.text import MIMEText

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


def reset_link(token: str) -> str:
    base = get_settings().auth.frontend_url.rstrip("/")
    return f"{base}/reset?token={token}"


def send_password_reset(email: str, token: str) -> bool:
    """Send (or log) a password-reset link. Returns True if actually emailed."""
    cfg = get_settings().auth
    link = reset_link(token)

    if not cfg.smtp_host:
        # Dev path: no SMTP — surface the link in logs so it's usable locally.
        logger.info("[dev] password reset for %s → %s", email, link)
        return False

    msg = MIMEText(
        f"Reset your VectorFlow password using the link below.\n\n{link}\n\n"
        "If you didn't request this, you can ignore this email."
    )
    msg["Subject"] = "Reset your VectorFlow password"
    msg["From"] = cfg.smtp_from or cfg.smtp_user or "no-reply@vectorflow.local"
    msg["To"] = email

    import smtplib

    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as server:
            server.starttls()
            if cfg.smtp_user and cfg.smtp_password:
                server.login(cfg.smtp_user, cfg.smtp_password)
            server.sendmail(msg["From"], [email], msg.as_string())
        logger.info("Sent password-reset email to %s", email)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Failed to send reset email to %s: %s", email, exc)
        return False

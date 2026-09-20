"""
pipeline/alerts.py
──────────────────
Deduplication + multi-channel alert delivery for CallGuard.

Channels (each independently behind config flags):
  • dashboard  — sets st.session_state flag for the UI banner (always on)
  • telegram   — requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  • email      — requires SMTP_HOST, SMTP_USER, SMTP_PASS, SMTP_TO
  • twilio     — requires TWILIO_ENABLED=true + Twilio credentials

Design rules:
  • Failure in one channel NEVER stops the others or crashes the app.
  • Per-category cooldown prevents duplicate alerts within ALERT_COOLDOWN_S.
  • All sent alerts are persisted via db.database.save_alert().
"""

from __future__ import annotations

import time
import threading
from datetime import datetime
from typing import Any

from config import cfg

# ── Deduplication state ───────────────────────────────────────────────────────
# {category: last_sent_timestamp}
_last_sent: dict[str, float] = {}
_lock = threading.Lock()


def _cooldown_ok(category: str) -> bool:
    with _lock:
        last = _last_sent.get(category, 0.0)
        if time.time() - last >= cfg.ALERT_COOLDOWN_S:
            _last_sent[category] = time.time()
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Channel implementations
# ─────────────────────────────────────────────────────────────────────────────

def _send_telegram(level: str, category: str, confidence: float, text: str) -> bool:
    if not cfg.TELEGRAM_BOT_TOKEN or not cfg.TELEGRAM_CHAT_ID:
        return False
    try:
        import requests
        emoji = "🚨" if level == "emergency" else "⚠️"
        ts    = datetime.utcnow().strftime("%H:%M:%S UTC")
        msg   = (
            f"{emoji} *CallGuard {level.upper()}*\n"
            f"Category: `{category}`\n"
            f"Confidence: `{confidence:.0%}`\n"
            f"Time: `{ts}`\n"
            f"Text: _{text[:280]}_"
        )
        resp = requests.post(
            f"https://api.telegram.org/bot{cfg.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id":    cfg.TELEGRAM_CHAT_ID,
                "text":       msg,
                "parse_mode": "Markdown",
            },
            timeout=8,
        )
        resp.raise_for_status()
        print(f"[alerts] Telegram sent ({level} / {category})")
        return True
    except Exception as exc:
        print(f"[alerts] Telegram FAILED: {exc}")
        return False


def _send_email(level: str, category: str, confidence: float, text: str) -> bool:
    if not cfg.SMTP_HOST or not cfg.SMTP_TO:
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = f"[CallGuard {level.upper()}] {category} detected"
        body = (
            f"CallGuard Alert\n"
            f"{'='*50}\n"
            f"Level      : {level.upper()}\n"
            f"Category   : {category}\n"
            f"Confidence : {confidence:.0%}\n"
            f"Time       : {ts}\n"
            f"\nFlagged text:\n{text}\n"
        )
        msg = MIMEMultipart()
        msg["From"]    = cfg.SMTP_FROM
        msg["To"]      = cfg.SMTP_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(cfg.SMTP_HOST, cfg.SMTP_PORT, timeout=10) as server:
            server.ehlo()
            if cfg.SMTP_PORT in (587, 465):
                server.starttls()
            if cfg.SMTP_USER and cfg.SMTP_PASS:
                server.login(cfg.SMTP_USER, cfg.SMTP_PASS)
            server.sendmail(cfg.SMTP_FROM, cfg.SMTP_TO.split(","), msg.as_string())

        print(f"[alerts] Email sent to {cfg.SMTP_TO} ({level} / {category})")
        return True
    except Exception as exc:
        print(f"[alerts] Email FAILED: {exc}")
        return False


def _send_twilio(level: str, category: str, confidence: float, text: str) -> bool:
    if not cfg.TWILIO_ENABLED:
        return False
    try:
        from twilio.rest import Client
        client = Client(cfg.TWILIO_ACCOUNT_SID, cfg.TWILIO_AUTH_TOKEN)
        body   = (
            f"[CallGuard {level.upper()}] {category} detected "
            f"({confidence:.0%} confidence): {text[:120]}"
        )
        client.messages.create(to=cfg.TWILIO_TO, from_=cfg.TWILIO_FROM, body=body)
        print(f"[alerts] Twilio SMS sent ({level} / {category})")
        return True
    except Exception as exc:
        print(f"[alerts] Twilio FAILED: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fire_alert(
    level: str,
    category: str,
    confidence: float,
    text: str,
    call_id: int | None = None,
    session_id: str | None = None,
    evidence_path: str | None = None,
    persist: bool = True,
) -> list[str]:
    """
    Fire an alert through all configured channels.

    Parameters
    ----------
    level          : "warning" | "emergency"
    category       : threat category string
    confidence     : 0.0 – 1.0
    text           : flagged segment text
    call_id        : DB call ID (file analysis) or None (live)
    session_id     : live session UUID or None
    evidence_path  : path to saved audio evidence clip or None
    persist        : save to DB alerts table (default True)

    Returns
    -------
    List of channel names that succeeded.
    """
    if not _cooldown_ok(category):
        print(f"[alerts] Cooldown active for {category} — alert suppressed")
        return []

    channels_sent: list[str] = []

    # ── Dashboard (always — sets Streamlit session state) ─────────────────
    try:
        import streamlit as st
        st.session_state["alert_level"]     = level
        st.session_state["alert_category"]  = category
        st.session_state["alert_confidence"]= confidence
        st.session_state["alert_text"]      = text
        st.session_state["alert_ts"]        = time.time()
        channels_sent.append("dashboard")
    except Exception:
        # Running outside Streamlit context (e.g. unit tests)
        channels_sent.append("dashboard")

    # ── Telegram ──────────────────────────────────────────────────────────
    if _send_telegram(level, category, confidence, text):
        channels_sent.append("telegram")

    # ── Email ─────────────────────────────────────────────────────────────
    if _send_email(level, category, confidence, text):
        channels_sent.append("email")

    # ── Twilio ────────────────────────────────────────────────────────────
    if _send_twilio(level, category, confidence, text):
        channels_sent.append("twilio")

    # ── Persist to DB ─────────────────────────────────────────────────────
    if persist:
        try:
            from db.database import get_session, save_alert
            with get_session() as s:
                save_alert(
                    s,
                    level         = level,
                    category      = category,
                    confidence    = confidence,
                    text          = text,
                    call_id       = call_id,
                    session_id    = session_id,
                    channels_sent = channels_sent,
                    evidence_path = evidence_path,
                )
        except Exception as exc:
            print(f"[alerts] DB persist FAILED: {exc}")

    print(f"[alerts] Fired {level} ({category} {confidence:.0%}) → {channels_sent}")
    return channels_sent


def send_test_alert() -> list[str]:
    """Send a test alert through all channels regardless of cooldown."""
    with _lock:
        _last_sent.clear()
    return fire_alert(
        level      = "warning",
        category   = "threat",
        confidence = 0.75,
        text       = "TEST ALERT — please ignore",
        persist    = True,
    )

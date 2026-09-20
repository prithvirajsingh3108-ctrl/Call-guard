"""
db/database.py
──────────────
Database engine, session management, CRUD helpers.

Migration policy
----------------
Every new nullable column is added by _run_migrations() via ALTER TABLE so
existing callguard.db files keep working without data loss.

Migrations applied
------------------
v2  call_summaries.risk_score            INTEGER DEFAULT 0
v3  segments: asr_text, asr_language, asr_confidence, no_speech_prob,
              normalized_text
v4  flags: keyword_score, semantic_score, emotion_score, llm_score,
           llm_reason, stage_details, severity, review_status
v5  alerts table (created by create_all if missing)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from db.models import (
    Base, Call, Segment, Flag, CallSummary, VoiceProfile, Alert,
)

load_dotenv()

# ── Engine ────────────────────────────────────────────────────────────────────
DATABASE_PATH = os.getenv("DATABASE_PATH", "callguard.db")
DATABASE_URL  = f"sqlite:///{DATABASE_PATH}"
_engine = sa.create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


# ── Migration helper ──────────────────────────────────────────────────────────

def _table_columns(conn: sa.Connection, table: str) -> set[str]:
    rows = conn.execute(sa.text(f"PRAGMA table_info({table})")).fetchall()
    return {row[1] for row in rows}


def _add_col_if_missing(
    conn: sa.Connection,
    table: str,
    column: str,
    coltype: str,
    default: str = "NULL",
) -> bool:
    """ALTER TABLE … ADD COLUMN if the column does not already exist.
    Returns True if the column was added."""
    if column not in _table_columns(conn, table):
        conn.execute(
            sa.text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype} DEFAULT {default}")
        )
        print(f"[db] Migration: {table}.{column} added ({coltype})")
        return True
    return False


def _run_migrations() -> None:
    with _engine.connect() as conn:
        # v2 — risk_score on call_summaries
        _add_col_if_missing(conn, "call_summaries", "risk_score", "INTEGER", "0")

        # v3 — ASR diagnostics on segments
        for col, typ in [
            ("asr_text",        "TEXT"),
            ("asr_language",    "TEXT"),
            ("asr_confidence",  "REAL"),
            ("no_speech_prob",  "REAL"),
            ("normalized_text", "TEXT"),
        ]:
            _add_col_if_missing(conn, "segments", col, typ)

        # v4 — stage scores + review on flags
        for col, typ, dflt in [
            ("keyword_score",  "REAL",    "NULL"),
            ("semantic_score", "REAL",    "NULL"),
            ("emotion_score",  "REAL",    "NULL"),
            ("llm_score",      "REAL",    "NULL"),
            ("llm_reason",     "TEXT",    "NULL"),
            ("stage_details",  "TEXT",    "NULL"),   # stored as JSON string
            ("severity",       "TEXT",    "'Medium'"),
            ("review_status",  "TEXT",    "'pending'"),
        ]:
            _add_col_if_missing(conn, "flags", col, typ, dflt)

        conn.commit()


# ── Public initialiser ────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables (if missing) then run incremental migrations."""
    Base.metadata.create_all(bind=_engine)
    _run_migrations()
    print(f"[db] Database ready: {Path(DATABASE_PATH).resolve()}")


# ── Session context manager ───────────────────────────────────────────────────

@contextmanager
def get_session():
    session = Session(bind=_engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Call helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_call(session: Session, filename: str, file_path: str | None = None) -> Call:
    call = Call(filename=filename, file_path=file_path, status="pending")
    session.add(call)
    session.flush()
    return call


def update_call_status(
    session: Session, call_id: int, status: str, error_msg: str | None = None
) -> None:
    call = session.get(Call, call_id)
    if call is None:
        raise ValueError(f"No call with id={call_id}")
    call.status = status
    if error_msg:
        call.error_msg = error_msg
    if status == "done":
        call.completed_at = datetime.utcnow()


def get_all_calls(session: Session) -> list[Call]:
    return session.execute(
        sa.select(Call).order_by(Call.created_at.desc())
    ).scalars().all()


def get_call_by_id(session: Session, call_id: int) -> Call | None:
    return session.get(Call, call_id)


def delete_call(session: Session, call_id: int) -> bool:
    call = session.get(Call, call_id)
    if call is None:
        return False
    session.delete(call)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Segment / Flag / Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_call_results(
    session: Session,
    call_id: int,
    enriched_segments: list[dict],
    summary: dict,
    duration_sec: float | None = None,
    risk_score: int = 0,
) -> None:
    """
    Persist all segments, flags and summary for a completed analysis.

    Accepts the enriched segment dicts produced by detector.detect_threats().
    New Phase-1/3 fields (asr_text, stage scores, etc.) are written when
    present in the dict; absent keys are stored as NULL so old pipelines
    continue to work.
    """
    call = session.get(Call, call_id)
    if call is None:
        raise ValueError(f"No call with id={call_id}")
    if duration_sec is not None:
        call.duration_sec = duration_sec

    for seg_data in enriched_segments:
        segment = Segment(
            call_id          = call_id,
            speaker          = seg_data.get("speaker_original", seg_data["speaker"]),
            text             = seg_data["text"],
            start_sec        = seg_data["start"],
            end_sec          = seg_data["end"],
            is_flagged       = seg_data.get("flag", False),
            matched_name     = seg_data.get("matched_name"),
            match_confidence = seg_data.get("match_confidence"),
            match_status     = seg_data.get("match_status", "not_run"),
            # Phase-1 ASR diagnostics
            asr_text         = seg_data.get("asr_text"),
            asr_language     = seg_data.get("asr_language"),
            asr_confidence   = seg_data.get("asr_confidence"),
            no_speech_prob   = seg_data.get("no_speech_prob"),
            normalized_text  = seg_data.get("normalized_text"),
        )
        session.add(segment)
        session.flush()

        if seg_data.get("flag"):
            flag = Flag(
                call_id        = call_id,
                segment_id     = segment.id,
                speaker        = seg_data["speaker"],
                matched_text   = seg_data["text"],
                matched_keyword= seg_data.get("matched_keyword", ""),
                category       = seg_data.get("category", ""),
                confidence     = seg_data.get("confidence", 0.0),
                timestamp_sec  = seg_data["start"],
                context_window = seg_data.get("context_window", []),
                # Phase-3 stage scores
                keyword_score  = seg_data.get("keyword_score"),
                semantic_score = seg_data.get("semantic_score"),
                emotion_score  = seg_data.get("emotion_score"),
                llm_score      = seg_data.get("llm_score"),
                llm_reason     = seg_data.get("llm_reason"),
                stage_details  = seg_data.get("stage_details"),
                severity       = seg_data.get("severity", "Medium"),
                review_status  = "pending",
            )
            session.add(flag)

    # Upsert CallSummary
    existing = session.execute(
        sa.select(CallSummary).where(CallSummary.call_id == call_id)
    ).scalar_one_or_none()
    if existing:
        session.delete(existing)
        session.flush()

    session.add(CallSummary(
        call_id            = call_id,
        total_segments     = summary["total_segments"],
        total_flags        = summary["total_flags"],
        by_category        = summary["by_category"],
        highest_confidence = summary["highest_confidence"],
        risk_score         = risk_score,
    ))


def get_segments_for_call(session: Session, call_id: int) -> list[Segment]:
    return session.execute(
        sa.select(Segment)
        .where(Segment.call_id == call_id)
        .order_by(Segment.start_sec)
    ).scalars().all()


def get_flags_for_call(session: Session, call_id: int) -> list[Flag]:
    return session.execute(
        sa.select(Flag)
        .where(Flag.call_id == call_id)
        .order_by(Flag.timestamp_sec)
    ).scalars().all()


def get_summary_for_call(session: Session, call_id: int) -> CallSummary | None:
    return session.execute(
        sa.select(CallSummary).where(CallSummary.call_id == call_id)
    ).scalar_one_or_none()


def update_flag_review(
    session: Session, flag_id: int, review_status: str
) -> Flag | None:
    """Set review_status on a flag: 'confirmed' | 'false_alarm'."""
    flag = session.get(Flag, flag_id)
    if flag:
        flag.review_status = review_status
    return flag


# ─────────────────────────────────────────────────────────────────────────────
# Alert helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_alert(
    session: Session,
    level: str,
    category: str,
    confidence: float,
    text: str,
    call_id: int | None = None,
    session_id: str | None = None,
    channels_sent: list[str] | None = None,
    evidence_path: str | None = None,
) -> Alert:
    alert = Alert(
        call_id       = call_id,
        session_id    = session_id,
        level         = level,
        category      = category,
        confidence    = confidence,
        text          = text,
        channels_sent = channels_sent or [],
        evidence_path = evidence_path,
    )
    session.add(alert)
    session.flush()
    return alert


def get_recent_alerts(session: Session, limit: int = 50) -> list[Alert]:
    return session.execute(
        sa.select(Alert).order_by(Alert.created_at.desc()).limit(limit)
    ).scalars().all()


# ─────────────────────────────────────────────────────────────────────────────
# Voice profile helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_voice_profile(
    session: Session, name: str, embedding: list[float], audio_file: str | None = None
) -> VoiceProfile:
    existing = session.execute(
        sa.select(VoiceProfile).where(VoiceProfile.name == name)
    ).scalar_one_or_none()
    if existing:
        session.delete(existing)
        session.flush()
    profile = VoiceProfile(name=name, embedding=embedding, audio_file=audio_file)
    session.add(profile)
    session.flush()
    return profile


def get_all_voice_profiles(session: Session) -> list[VoiceProfile]:
    return session.execute(
        sa.select(VoiceProfile).order_by(VoiceProfile.date_added.desc())
    ).scalars().all()


def delete_voice_profile(session: Session, profile_id: int) -> bool:
    profile = session.get(VoiceProfile, profile_id)
    if profile is None:
        return False
    session.delete(profile)
    return True


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising database…")
    init_db()
    with get_session() as s:
        call = create_call(s, filename="smoke_test.wav")
        print(f"Created: {call}")
    with get_session() as s:
        calls = get_all_calls(s)
        print(f"Calls in DB: {len(calls)}")
    print("Database test complete.")

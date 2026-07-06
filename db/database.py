"""
database.py
───────────
Database connection, session management, and helper functions for
reading/writing CallGuard data.

The SQLite file path is read from DATABASE_PATH in .env (default: callguard.db).

Typical usage:
    from db.database import get_session, save_call_results, get_all_calls

    with get_session() as session:
        calls = get_all_calls(session)
"""

import os
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from db.models import Base, Call, Segment, Flag, CallSummary, VoiceProfile

load_dotenv()

# ── Engine setup ──────────────────────────────────────────────────────────────
DATABASE_PATH = os.getenv("DATABASE_PATH", "callguard.db")
DATABASE_URL  = f"sqlite:///{DATABASE_PATH}"

# echo=False hides SQL logs; set to True for debugging
_engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def init_db():
    """Create all tables if they don't already exist. Safe to call on every start."""
    Base.metadata.create_all(bind=_engine)
    print(f"[db] Database ready: {Path(DATABASE_PATH).resolve()}")


@contextmanager
def get_session():
    """
    Context manager that yields a SQLAlchemy Session and commits on exit.
    Rolls back on exception.

    Usage:
        with get_session() as session:
            session.add(some_object)
    """
    session = Session(bind=_engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Write helpers ─────────────────────────────────────────────────────────────

def create_call(session: Session, filename: str, file_path: str = None) -> Call:
    """
    Insert a new Call record with status='pending' and return it.
    """
    call = Call(
        filename  = filename,
        file_path = file_path,
        status    = "pending",
    )
    session.add(call)
    session.flush()   # assigns call.id without a full commit
    return call


def update_call_status(session: Session, call_id: int, status: str, error_msg: str = None):
    """Update the status (and optionally error message) of a Call."""
    call = session.get(Call, call_id)
    if call is None:
        raise ValueError(f"No call with id={call_id}")
    call.status = status
    if error_msg:
        call.error_msg = error_msg
    if status == "done":
        call.completed_at = datetime.utcnow()


def save_call_results(
    session: Session,
    call_id: int,
    enriched_segments: list[dict],
    summary: dict,
    duration_sec: float = None,
) -> None:
    """
    Persist transcript segments, flags, and call summary to the database.

    Args:
        call_id:           ID of the Call row.
        enriched_segments: Output of detector.detect_threats() — each dict
                           has speaker/text/start/end + flag/category/confidence/...
        summary:           Output of detector.summarize_flags().
        duration_sec:      Total audio duration (optional).
    """
    # Update Call duration + status
    call = session.get(Call, call_id)
    if call is None:
        raise ValueError(f"No call with id={call_id}")
    if duration_sec is not None:
        call.duration_sec = duration_sec

    # Insert each segment
    for seg_data in enriched_segments:
        segment = Segment(
            call_id    = call_id,
            speaker    = seg_data["speaker"],
            text       = seg_data["text"],
            start_sec  = seg_data["start"],
            end_sec    = seg_data["end"],
            is_flagged = seg_data.get("flag", False),
        )
        session.add(segment)
        session.flush()   # get segment.id

        # Insert Flag row if this segment was flagged
        if seg_data.get("flag"):
            flag = Flag(
                call_id         = call_id,
                segment_id      = segment.id,
                speaker         = seg_data["speaker"],
                matched_text    = seg_data["text"],
                matched_keyword = seg_data.get("matched_keyword", ""),
                category        = seg_data.get("category", ""),
                confidence      = seg_data.get("confidence", 0.0),
                timestamp_sec   = seg_data["start"],
                context_window  = seg_data.get("context_window", []),
            )
            session.add(flag)

    # Insert / replace CallSummary
    existing_summary = session.execute(
        select(CallSummary).where(CallSummary.call_id == call_id)
    ).scalar_one_or_none()

    if existing_summary:
        # Overwrite if re-processing
        session.delete(existing_summary)
        session.flush()

    call_summary = CallSummary(
        call_id            = call_id,
        total_segments     = summary["total_segments"],
        total_flags        = summary["total_flags"],
        by_category        = summary["by_category"],
        highest_confidence = summary["highest_confidence"],
    )
    session.add(call_summary)


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_all_calls(session: Session) -> list[Call]:
    """Return all calls ordered by most recent first."""
    return session.execute(
        select(Call).order_by(Call.created_at.desc())
    ).scalars().all()


def get_call_by_id(session: Session, call_id: int) -> Call | None:
    """Return a single Call by its id, or None if not found."""
    return session.get(Call, call_id)


def get_segments_for_call(session: Session, call_id: int) -> list[Segment]:
    """Return all segments for a call, ordered by start time."""
    return session.execute(
        select(Segment)
        .where(Segment.call_id == call_id)
        .order_by(Segment.start_sec)
    ).scalars().all()


def get_flags_for_call(session: Session, call_id: int) -> list[Flag]:
    """Return all flags for a call, ordered by timestamp."""
    return session.execute(
        select(Flag)
        .where(Flag.call_id == call_id)
        .order_by(Flag.timestamp_sec)
    ).scalars().all()


def get_summary_for_call(session: Session, call_id: int) -> CallSummary | None:
    """Return the CallSummary for a call, or None if not yet computed."""
    return session.execute(
        select(CallSummary).where(CallSummary.call_id == call_id)
    ).scalar_one_or_none()


def delete_call(session: Session, call_id: int) -> bool:
    """
    Delete a call and all related segments/flags/summary (cascade).
    Returns True if found and deleted, False if not found.
    """
    call = session.get(Call, call_id)
    if call is None:
        return False
    session.delete(call)
    return True


# ── Voice profile helpers ─────────────────────────────────────────────────────

def save_voice_profile(session: Session, name: str, embedding: list[float], audio_file: str = None) -> VoiceProfile:
    """Insert or replace a voice profile for a named person."""
    # Remove existing profile with same name if present
    existing = session.execute(
        select(VoiceProfile).where(VoiceProfile.name == name)
    ).scalar_one_or_none()
    if existing:
        session.delete(existing)
        session.flush()

    profile = VoiceProfile(
        name       = name,
        embedding  = embedding,
        audio_file = audio_file,
    )
    session.add(profile)
    session.flush()
    return profile


def get_all_voice_profiles(session: Session) -> list[VoiceProfile]:
    """Return all enrolled voice profiles."""
    return session.execute(
        select(VoiceProfile).order_by(VoiceProfile.date_added.desc())
    ).scalars().all()


def delete_voice_profile(session: Session, profile_id: int) -> bool:
    """Delete a voice profile by id. Returns True if deleted."""
    profile = session.get(VoiceProfile, profile_id)
    if profile is None:
        return False
    session.delete(profile)
    return True


# ── CLI: quick schema test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising database...")
    init_db()

    # Insert a sample call to verify everything works
    with get_session() as session:
        call = create_call(session, filename="test_call.wav", file_path="sample_audio/test_call.wav")
        print(f"Created: {call}")

    with get_session() as session:
        calls = get_all_calls(session)
        print(f"\nAll calls in DB ({len(calls)}):")
        for c in calls:
            print(f"  {c}")

    print("\nDatabase test complete. Check callguard.db to verify.")

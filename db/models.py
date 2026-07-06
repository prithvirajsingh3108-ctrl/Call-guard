"""
models.py
─────────
SQLAlchemy ORM models for the CallGuard database.

Tables
------
Call          One row per uploaded/processed audio file.
Segment       One row per transcript segment (speaker turn).
Flag          One row per detected threat in a segment.
CallSummary   One row per call — aggregate statistics.

All tables are created automatically by database.py on first run.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, Boolean,
    DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared base class for all models."""
    pass


class VoiceProfile(Base):
    """
    Stores a named person's voice embedding for cross-call speaker recognition.

    embedding is stored as a JSON array of floats (256-dim resemblyzer vector).
    """
    __tablename__ = "voice_profiles"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(256), nullable=False, unique=True)
    embedding  = Column(JSON, nullable=False)   # list of 256 floats
    audio_file = Column(String(512), nullable=True)  # original sample filename
    date_added = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<VoiceProfile id={self.id} name='{self.name}'>"


class Call(Base):
    """
    Represents a processed audio call.

    created_at is set automatically to the current UTC time.
    """
    __tablename__ = "calls"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    filename     = Column(String(512), nullable=False)   # original filename
    file_path    = Column(String(1024), nullable=True)   # where it was stored
    duration_sec = Column(Float, nullable=True)          # audio duration in seconds
    language     = Column(String(16), nullable=True)     # detected language code
    status       = Column(
        String(32), nullable=False, default="pending"
        # pending | transcribing | analyzing | done | error
    )
    error_msg    = Column(Text, nullable=True)           # filled if status=error
    created_at   = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships — cascade deletes so removing a Call cleans everything up
    segments = relationship("Segment", back_populates="call", cascade="all, delete-orphan")
    summary  = relationship("CallSummary", back_populates="call", uselist=False,
                            cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Call id={self.id} file='{self.filename}' status='{self.status}'>"


class Segment(Base):
    """
    One speaker turn from the transcript.

    Stores the raw text plus detection results for this segment.
    """
    __tablename__ = "segments"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    call_id      = Column(Integer, ForeignKey("calls.id"), nullable=False)
    speaker      = Column(String(64), nullable=False)   # e.g. "SPEAKER_00"
    text         = Column(Text, nullable=False)
    start_sec    = Column(Float, nullable=False)
    end_sec      = Column(Float, nullable=False)
    is_flagged   = Column(Boolean, default=False, nullable=False)

    call  = relationship("Call", back_populates="segments")
    flags = relationship("Flag", back_populates="segment", cascade="all, delete-orphan")

    def __repr__(self):
        snippet = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"<Segment id={self.id} speaker='{self.speaker}' flagged={self.is_flagged} '{snippet}'>"


class Flag(Base):
    """
    A single threat detection hit within a segment.

    A segment may have multiple flags (e.g., matches both 'threat'
    and 'harm_planning').

    context_window is stored as a JSON array of {"speaker", "text"} dicts
    so we can replay the conversation context that led to the flag.
    """
    __tablename__ = "flags"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    call_id          = Column(Integer, ForeignKey("calls.id"), nullable=False)
    segment_id       = Column(Integer, ForeignKey("segments.id"), nullable=False)
    speaker          = Column(String(64), nullable=False)
    matched_text     = Column(Text, nullable=False)       # text of the flagged segment
    matched_keyword  = Column(String(256), nullable=False)
    category         = Column(String(64), nullable=False)  # threat/abuse/harm_planning/disaster
    confidence       = Column(Float, nullable=False)       # 0.0 – 1.0
    timestamp_sec    = Column(Float, nullable=False)       # start time in the call
    context_window   = Column(JSON, nullable=True)         # list of prior segments
    created_at       = Column(DateTime, default=datetime.utcnow)

    segment = relationship("Segment", back_populates="flags")

    def __repr__(self):
        return (
            f"<Flag id={self.id} cat='{self.category}' "
            f"conf={self.confidence:.2f} at {self.timestamp_sec}s>"
        )


class CallSummary(Base):
    """
    Aggregate statistics for a completed call.

    by_category is stored as JSON: {"threat": 2, "abuse": 1, ...}
    """
    __tablename__ = "call_summaries"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    call_id             = Column(Integer, ForeignKey("calls.id"), nullable=False, unique=True)
    total_segments      = Column(Integer, nullable=False, default=0)
    total_flags         = Column(Integer, nullable=False, default=0)
    by_category         = Column(JSON, nullable=True)   # {"threat": 2, ...}
    highest_confidence  = Column(Float, nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    call = relationship("Call", back_populates="summary")

    def __repr__(self):
        return (
            f"<CallSummary call_id={self.call_id} "
            f"flags={self.total_flags}/{self.total_segments}>"
        )

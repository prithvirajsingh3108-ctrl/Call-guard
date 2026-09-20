"""
db/models.py
────────────
SQLAlchemy ORM models for the CallGuard database.

Tables
------
VoiceProfile  Enrolled speaker embeddings.
Call          One row per processed audio file.
Segment       One row per transcript segment (speaker turn).
Flag          One row per detected threat in a segment.
CallSummary   Aggregate statistics per call.
Alert         Emergency / warning alerts fired during live or file analysis.

Migration policy
----------------
New nullable columns are added via ALTER TABLE in init_db() so existing
callguard.db files keep working without a full rebuild.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, Boolean,
    DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────

class VoiceProfile(Base):
    """Named speaker voice embedding (256-dim resemblyzer)."""
    __tablename__ = "voice_profiles"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(256), nullable=False, unique=True)
    embedding  = Column(JSON, nullable=False)
    audio_file = Column(String(512), nullable=True)
    date_added = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<VoiceProfile id={self.id} name='{self.name}'>"


# ─────────────────────────────────────────────────────────────────────────────

class Call(Base):
    """One uploaded/processed audio recording."""
    __tablename__ = "calls"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    filename     = Column(String(512), nullable=False)
    file_path    = Column(String(1024), nullable=True)
    duration_sec = Column(Float, nullable=True)
    language     = Column(String(16), nullable=True)   # detected language code
    status       = Column(String(32), nullable=False, default="pending")
    # pending | transcribing | analyzing | done | error
    error_msg    = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    segments = relationship("Segment", back_populates="call", cascade="all, delete-orphan")
    summary  = relationship("CallSummary", back_populates="call", uselist=False,
                            cascade="all, delete-orphan")
    alerts   = relationship("Alert", back_populates="call", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Call id={self.id} file='{self.filename}' status='{self.status}'>"


# ─────────────────────────────────────────────────────────────────────────────

class Segment(Base):
    """
    One speaker turn from the transcript.

    Phase-1 additions (nullable, added via ALTER TABLE migration):
      asr_text        — raw ASR output before any normalisation
      asr_language    — per-segment detected language code
      asr_confidence  — avg_logprob from Whisper (negative float; higher = better)
      no_speech_prob  — Whisper's no-speech probability (0-1; lower = more speech)
      normalized_text — text after pipeline/normalize.py processing
    """
    __tablename__ = "segments"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    call_id          = Column(Integer, ForeignKey("calls.id"), nullable=False)
    speaker          = Column(String(64), nullable=False)
    text             = Column(Text, nullable=False)
    start_sec        = Column(Float, nullable=False)
    end_sec          = Column(Float, nullable=False)
    is_flagged       = Column(Boolean, default=False, nullable=False)
    matched_name     = Column(String(256), nullable=True)
    match_confidence = Column(Float, nullable=True)
    match_status     = Column(String(32), nullable=True)

    # Phase 1 ASR diagnostics (nullable — added by migration)
    asr_text        = Column(Text, nullable=True)
    asr_language    = Column(String(16), nullable=True)
    asr_confidence  = Column(Float, nullable=True)   # avg_logprob
    no_speech_prob  = Column(Float, nullable=True)
    normalized_text = Column(Text, nullable=True)

    call  = relationship("Call", back_populates="segments")
    flags = relationship("Flag", back_populates="segment", cascade="all, delete-orphan")

    def __repr__(self):
        snippet = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"<Segment id={self.id} speaker='{self.speaker}' flagged={self.is_flagged} '{snippet}'>"


# ─────────────────────────────────────────────────────────────────────────────

class Flag(Base):
    """
    One threat detection hit within a segment.

    Phase-1/3 additions (nullable, added via ALTER TABLE migration):
      keyword_score   — raw rapidfuzz score (0-1)
      semantic_score  — cosine similarity score (0-1)
      emotion_score   — prosody/emotion bonus (0-1)
      llm_score       — LLM judge confidence (0-1); None if not called
      llm_reason      — one-line explanation from LLM
      stage_details   — JSON dict with all intermediate scores and decisions
      severity        — Low | Medium | High
      review_status   — pending | confirmed | false_alarm
    """
    __tablename__ = "flags"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    call_id          = Column(Integer, ForeignKey("calls.id"), nullable=False)
    segment_id       = Column(Integer, ForeignKey("segments.id"), nullable=False)
    speaker          = Column(String(64), nullable=False)
    matched_text     = Column(Text, nullable=False)
    matched_keyword  = Column(String(256), nullable=False)
    category         = Column(String(64), nullable=False)
    confidence       = Column(Float, nullable=False)
    timestamp_sec    = Column(Float, nullable=False)
    context_window   = Column(JSON, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    # Phase 3 stage scores (nullable — added by migration)
    keyword_score  = Column(Float, nullable=True)
    semantic_score = Column(Float, nullable=True)
    emotion_score  = Column(Float, nullable=True)
    llm_score      = Column(Float, nullable=True)
    llm_reason     = Column(Text, nullable=True)
    stage_details  = Column(JSON, nullable=True)   # full per-stage breakdown
    severity       = Column(String(16), nullable=True, default="Medium")
    # "Low" | "Medium" | "High"

    # Phase 3 review (nullable — added by migration)
    review_status  = Column(String(32), nullable=True, default="pending")
    # "pending" | "confirmed" | "false_alarm"

    segment = relationship("Segment", back_populates="flags")

    def __repr__(self):
        return (
            f"<Flag id={self.id} cat='{self.category}' "
            f"conf={self.confidence:.2f} sev={self.severity} "
            f"review={self.review_status} at {self.timestamp_sec}s>"
        )


# ─────────────────────────────────────────────────────────────────────────────

class CallSummary(Base):
    """Aggregate statistics for a completed call."""
    __tablename__ = "call_summaries"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    call_id             = Column(Integer, ForeignKey("calls.id"), nullable=False, unique=True)
    total_segments      = Column(Integer, nullable=False, default=0)
    total_flags         = Column(Integer, nullable=False, default=0)
    by_category         = Column(JSON, nullable=True)
    highest_confidence  = Column(Float, nullable=True)
    risk_score          = Column(Integer, nullable=True, default=0)   # 0–100
    created_at          = Column(DateTime, default=datetime.utcnow)

    call = relationship("Call", back_populates="summary")

    def __repr__(self):
        return (
            f"<CallSummary call_id={self.call_id} "
            f"flags={self.total_flags}/{self.total_segments} "
            f"risk={self.risk_score}>"
        )


# ─────────────────────────────────────────────────────────────────────────────

class Alert(Base):
    """
    Emergency or warning alert fired by the live monitor or file analysis.

    channels_sent: JSON list of channel names that successfully delivered,
                   e.g. ["dashboard", "telegram"].
    evidence_path: path to the saved 30-second audio clip (live mode).
    """
    __tablename__ = "alerts"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    call_id        = Column(Integer, ForeignKey("calls.id"), nullable=True)
    # nullable — live sessions may not have a persisted Call row yet
    session_id     = Column(String(64), nullable=True)   # live session UUID
    level          = Column(String(16), nullable=False, default="warning")
    # "warning" | "emergency"
    category       = Column(String(64), nullable=False)
    confidence     = Column(Float, nullable=False)
    text           = Column(Text, nullable=False)         # flagged segment text
    channels_sent  = Column(JSON, nullable=True)          # ["dashboard","telegram",…]
    evidence_path  = Column(String(1024), nullable=True)  # saved audio clip
    created_at     = Column(DateTime, default=datetime.utcnow)

    call = relationship("Call", back_populates="alerts")

    def __repr__(self):
        return (
            f"<Alert id={self.id} level='{self.level}' "
            f"cat='{self.category}' conf={self.confidence:.2f}>"
        )

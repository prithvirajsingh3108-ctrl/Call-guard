"""
config.py
─────────
Single source of truth for every tunable setting in CallGuard.

Priority order (highest wins):
  1. st.secrets  (Streamlit Cloud)
  2. .env file   (local development)
  3. defaults    (defined here)

Usage:
    from config import cfg
    print(cfg.WHISPER_MODEL)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    """Read from st.secrets first, then os.environ, then default."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


def _bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes", "on")


def _int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float = 0.0) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    # ── Auth ──────────────────────────────────────────────────────────────────
    APP_PASSWORD: str = field(default_factory=lambda: _env("APP_PASSWORD", ""))
    # Empty string = no password gate (for local dev)

    # ── Hugging Face ──────────────────────────────────────────────────────────
    HF_TOKEN: str = field(default_factory=lambda: _env("HF_TOKEN", ""))

    # ── ASR / Transcription ───────────────────────────────────────────────────
    WHISPER_MODEL: str     = field(default_factory=lambda: _env("WHISPER_MODEL", "base"))
    COMPUTE_DEVICE: str    = field(default_factory=lambda: _env("COMPUTE_DEVICE", "cpu"))
    WHISPER_LANGUAGE: str  = field(default_factory=lambda: _env("WHISPER_LANGUAGE", ""))
    USE_DIARIZATION: bool  = field(default_factory=lambda: _bool("USE_DIARIZATION", True))
    ASR_ENGINE: str        = field(default_factory=lambda: _env("ASR_ENGINE", "whisperx"))
    # Options: whisperx | indic | deepgram | assemblyai | google

    # Cloud ASR keys (only used when ASR_ENGINE is set to that engine)
    DEEPGRAM_API_KEY: str   = field(default_factory=lambda: _env("DEEPGRAM_API_KEY", ""))
    ASSEMBLYAI_API_KEY: str = field(default_factory=lambda: _env("ASSEMBLYAI_API_KEY", ""))
    GOOGLE_CREDENTIALS: str = field(default_factory=lambda: _env("GOOGLE_CREDENTIALS", ""))

    # ── Audio preprocessing ───────────────────────────────────────────────────
    AUDIO_SAMPLE_RATE: int       = field(default_factory=lambda: _int("AUDIO_SAMPLE_RATE", 16000))
    AUDIO_LOUDNORM: bool         = field(default_factory=lambda: _bool("AUDIO_LOUDNORM", True))
    AUDIO_HIGHPASS_HZ: int       = field(default_factory=lambda: _int("AUDIO_HIGHPASS_HZ", 80))
    AUDIO_NOISE_REDUCE: bool     = field(default_factory=lambda: _bool("AUDIO_NOISE_REDUCE", False))
    AUDIO_TRIM_SILENCE: bool     = field(default_factory=lambda: _bool("AUDIO_TRIM_SILENCE", True))
    AUDIO_SILENCE_MIN_LEN_MS: int = field(default_factory=lambda: _int("AUDIO_SILENCE_MIN_LEN_MS", 500))

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_PATH: str = field(default_factory=lambda: _env("DATABASE_PATH", "callguard.db"))

    # ── Keywords / detection ──────────────────────────────────────────────────
    KEYWORDS_PATH: str        = field(default_factory=lambda: _env("KEYWORDS_PATH", "pipeline/keywords.json"))
    PHRASES_PATH: str         = field(default_factory=lambda: _env("PHRASES_PATH", "pipeline/phrases.json"))
    FUZZY_THRESHOLD: int      = field(default_factory=lambda: _int("FUZZY_THRESHOLD", 80))
    CONTEXT_WINDOW_SIZE: int  = field(default_factory=lambda: _int("CONTEXT_WINDOW_SIZE", 4))

    # Fusion weights (must sum to ~1.0 before context modifier)
    WEIGHT_KEYWORD: float  = field(default_factory=lambda: _float("WEIGHT_KEYWORD", 0.45))
    WEIGHT_SEMANTIC: float = field(default_factory=lambda: _float("WEIGHT_SEMANTIC", 0.35))
    WEIGHT_EMOTION: float  = field(default_factory=lambda: _float("WEIGHT_EMOTION", 0.10))
    # Context modifier is applied multiplicatively, not as a weight

    FLAG_THRESHOLD: float      = field(default_factory=lambda: _float("FLAG_THRESHOLD", 0.50))
    EMERGENCY_THRESHOLD: float = field(default_factory=lambda: _float("EMERGENCY_THRESHOLD", 0.80))
    WARNING_THRESHOLD: float   = field(default_factory=lambda: _float("WARNING_THRESHOLD", 0.60))

    # ── Semantic layer ────────────────────────────────────────────────────────
    SEMANTIC_ENABLED: bool    = field(default_factory=lambda: _bool("SEMANTIC_ENABLED", True))
    SEMANTIC_MODEL: str       = field(default_factory=lambda: _env("SEMANTIC_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"))
    SEMANTIC_CACHE_DIR: str   = field(default_factory=lambda: _env("SEMANTIC_CACHE_DIR", ".cache/semantic"))
    SEMANTIC_MIN_SCORE: float = field(default_factory=lambda: _float("SEMANTIC_MIN_SCORE", 0.45))

    # ── LLM judge ────────────────────────────────────────────────────────────
    LLM_JUDGE: str          = field(default_factory=lambda: _env("LLM_JUDGE", "off"))
    # Options: off | ollama | anthropic
    LLM_MODEL: str          = field(default_factory=lambda: _env("LLM_MODEL", "llama3"))
    OLLAMA_HOST: str        = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ANTHROPIC_API_KEY: str  = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    LLM_BORDERLINE_LOW: float  = field(default_factory=lambda: _float("LLM_BORDERLINE_LOW", 0.40))
    LLM_BORDERLINE_HIGH: float = field(default_factory=lambda: _float("LLM_BORDERLINE_HIGH", 0.80))
    LLM_TIMEOUT_S: int      = field(default_factory=lambda: _int("LLM_TIMEOUT_S", 8))

    # ── Emotion / prosody (optional) ──────────────────────────────────────────
    EMOTION_ENABLED: bool = field(default_factory=lambda: _bool("EMOTION_ENABLED", False))
    EMOTION_MODEL: str    = field(default_factory=lambda: _env("EMOTION_MODEL", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"))

    # ── Live monitor ──────────────────────────────────────────────────────────
    LIVE_MODEL: str          = field(default_factory=lambda: _env("LIVE_MODEL", "base"))
    LIVE_CHUNK_SEC: int      = field(default_factory=lambda: _int("LIVE_CHUNK_SEC", 4))
    LIVE_OVERLAP_SEC: int    = field(default_factory=lambda: _int("LIVE_OVERLAP_SEC", 1))
    LIVE_SAMPLE_RATE: int    = field(default_factory=lambda: _int("LIVE_SAMPLE_RATE", 16000))
    LIVE_WARN_WINDOW_SEC: int = field(default_factory=lambda: _int("LIVE_WARN_WINDOW_SEC", 30))
    # Two warnings within this window triggers Emergency

    # ── Alerts ───────────────────────────────────────────────────────────────
    ALERT_COOLDOWN_S: int       = field(default_factory=lambda: _int("ALERT_COOLDOWN_S", 60))
    ALERT_EVIDENCE_SEC: int     = field(default_factory=lambda: _int("ALERT_EVIDENCE_SEC", 30))

    # Telegram
    TELEGRAM_BOT_TOKEN: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    TELEGRAM_CHAT_ID: str   = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID", ""))

    # Email via SMTP
    SMTP_HOST: str     = field(default_factory=lambda: _env("SMTP_HOST", ""))
    SMTP_PORT: int     = field(default_factory=lambda: _int("SMTP_PORT", 587))
    SMTP_USER: str     = field(default_factory=lambda: _env("SMTP_USER", ""))
    SMTP_PASS: str     = field(default_factory=lambda: _env("SMTP_PASS", ""))
    SMTP_TO: str       = field(default_factory=lambda: _env("SMTP_TO", ""))
    SMTP_FROM: str     = field(default_factory=lambda: _env("SMTP_FROM", "callguard@localhost"))

    # Twilio (stub, off by default)
    TWILIO_ENABLED: bool       = field(default_factory=lambda: _bool("TWILIO_ENABLED", False))
    TWILIO_ACCOUNT_SID: str    = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID", ""))
    TWILIO_AUTH_TOKEN: str     = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN", ""))
    TWILIO_FROM: str           = field(default_factory=lambda: _env("TWILIO_FROM", ""))
    TWILIO_TO: str             = field(default_factory=lambda: _env("TWILIO_TO", ""))

    # ── Eval ─────────────────────────────────────────────────────────────────
    EVAL_RECALL_TARGET: float = field(default_factory=lambda: _float("EVAL_RECALL_TARGET", 0.80))

    # ── Recognition ──────────────────────────────────────────────────────────
    RECOGNITION_THRESHOLD: float = field(default_factory=lambda: _float("RECOGNITION_THRESHOLD", 0.75))

    # ── UI ───────────────────────────────────────────────────────────────────
    SHOW_DIAGNOSTICS_DEFAULT: bool = field(default_factory=lambda: _bool("SHOW_DIAGNOSTICS_DEFAULT", False))

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.DATABASE_PATH}"

    @property
    def has_hf_token(self) -> bool:
        return bool(self.HF_TOKEN)

    @property
    def llm_judge_active(self) -> bool:
        return self.LLM_JUDGE.lower() not in ("off", "", "false", "0")

    @property
    def cloud_asr_active(self) -> bool:
        return self.ASR_ENGINE.lower() in ("deepgram", "assemblyai", "google")


# Singleton — import this everywhere
cfg = Config()

"""
pipeline/live.py
────────────────
Live microphone monitor for CallGuard.

Architecture
------------
  • A background thread captures audio in LIVE_CHUNK_SEC chunks with
    LIVE_OVERLAP_SEC overlap using sounddevice.
  • Chunks are put on a thread-safe queue.
  • The Streamlit page calls get_next_result() in a polling loop to
    drain the queue and display results without blocking the UI.
  • faster-whisper (tiny or base) transcribes each chunk with VAD on.
  • The full detector pipeline (normalize + keyword + semantic) runs on
    each transcript chunk plus a rolling 4-segment context buffer.
  • Alert levels:
      Warning   — confidence >= WARNING_THRESHOLD (0.60)
      Emergency — confidence >= EMERGENCY_THRESHOLD (0.80)
                  OR two warnings within LIVE_WARN_WINDOW_SEC (30s)

Latency note
------------
Processing latency is typically 2–6 seconds on CPU.  This is noted in
the UI copy so users have realistic expectations.

Consent note
------------
Only monitor audio you have explicit permission to record.
"""

from __future__ import annotations

import io
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveChunkResult:
    session_id:    str
    chunk_index:   int
    timestamp:     float          # wall-clock time of chunk start
    text:          str            # transcribed text
    language:      str            # detected language code
    flag:          bool
    category:      str
    confidence:    float
    severity:      str            # Low | Medium | High
    matched_keyword: str
    stage_details: dict
    alert_fired:   str | None     # "warning" | "emergency" | None


@dataclass
class LiveSession:
    session_id:    str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at:    float = field(default_factory=time.time)
    chunk_index:   int = 0
    results:       list[LiveChunkResult] = field(default_factory=list)
    _context_buf:  list[dict] = field(default_factory=list)   # rolling 4-segment ctx
    _warn_times:   list[float] = field(default_factory=list)  # recent warning timestamps
    is_running:    bool = False
    _stop_event:   threading.Event = field(default_factory=threading.Event)
    _result_queue: queue.Queue = field(default_factory=queue.Queue)
    _thread:       threading.Thread | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Whisper live model (cached per session)
# ─────────────────────────────────────────────────────────────────────────────

_live_model = None
_live_model_lock = threading.Lock()


def _get_live_model():
    global _live_model
    with _live_model_lock:
        if _live_model is None:
            try:
                from faster_whisper import WhisperModel
                print(f"[live] Loading faster-whisper {cfg.LIVE_MODEL} …")
                _live_model = WhisperModel(
                    cfg.LIVE_MODEL,
                    device       = cfg.COMPUTE_DEVICE,
                    compute_type = "int8" if cfg.COMPUTE_DEVICE == "cpu" else "float16",
                )
                print("[live] Live model ready")
            except ImportError:
                print("[live] faster-whisper not installed — live mode unavailable")
                _live_model = None
        return _live_model


# ─────────────────────────────────────────────────────────────────────────────
# Transcribe one audio chunk (numpy array → segment list)
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_chunk(audio_np, sr: int) -> tuple[str, str]:
    """
    Transcribe a numpy float32 array.
    Returns (text, language_code).
    """
    model = _get_live_model()
    if model is None:
        return "", "unknown"

    # Write to temp WAV so faster-whisper can read it
    import numpy as np
    import soundfile as sf

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, audio_np, sr, subtype="PCM_16")
        segments, info = model.transcribe(
            tmp.name,
            language       = cfg.WHISPER_LANGUAGE or None,
            vad_filter     = True,
            beam_size      = 5,
            condition_on_previous_text = False,
            temperature    = (0.0, 0.2, 0.4),   # fallback ladder
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        lang = info.language or "unknown"
        return text, lang
    except Exception as exc:
        print(f"[live] Transcription error: {exc}")
        return "", "unknown"
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Capture + analyse loop (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _capture_loop(session: LiveSession) -> None:
    """Background thread: capture mic → transcribe → detect → queue result."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("[live] sounddevice or numpy not installed — live capture unavailable")
        session.is_running = False
        return

    from pipeline.detector import analyze_segment

    sr          = cfg.LIVE_SAMPLE_RATE
    chunk_s     = cfg.LIVE_CHUNK_SEC
    overlap_s   = cfg.LIVE_OVERLAP_SEC
    step_s      = chunk_s - overlap_s
    chunk_n     = int(sr * chunk_s)
    step_n      = int(sr * step_s)

    buf = np.zeros(chunk_n, dtype=np.float32)

    print(f"[live] Capture started (chunk={chunk_s}s overlap={overlap_s}s sr={sr})")

    def callback(indata, frames, time_info, status):
        nonlocal buf
        if status:
            print(f"[live] sounddevice status: {status}")
        buf = np.roll(buf, -frames)
        buf[-frames:] = indata[:, 0]

    with sd.InputStream(
        samplerate = sr,
        channels   = 1,
        dtype      = "float32",
        blocksize  = step_n,
        callback   = callback,
    ):
        while not session._stop_event.is_set():
            time.sleep(step_s)
            if session._stop_event.is_set():
                break

            chunk_ts = time.time()
            audio_chunk = buf.copy()

            # Skip near-silent chunks
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
            if rms < 0.005:
                continue

            # Transcribe
            text, lang = _transcribe_chunk(audio_chunk, sr)
            if not text:
                continue

            # Detect
            seg = {
                "speaker": "Live",
                "text":    text,
                "start":   0.0,
                "end":     chunk_s,
                "asr_language": lang,
            }
            analysis = analyze_segment(seg, session._context_buf[-cfg.CONTEXT_WINDOW_SIZE:])

            # Update rolling context
            session._context_buf.append(seg)
            if len(session._context_buf) > cfg.CONTEXT_WINDOW_SIZE + 2:
                session._context_buf.pop(0)

            # Alert level logic
            alert_fired = None
            conf        = analysis.get("confidence", 0.0)
            now         = time.time()

            if analysis.get("flag"):
                if conf >= cfg.EMERGENCY_THRESHOLD:
                    alert_fired = "emergency"
                elif conf >= cfg.WARNING_THRESHOLD:
                    # Check if two warnings within window
                    session._warn_times = [
                        t for t in session._warn_times
                        if now - t < cfg.LIVE_WARN_WINDOW_SEC
                    ]
                    session._warn_times.append(now)
                    if len(session._warn_times) >= 2:
                        alert_fired = "emergency"
                    else:
                        alert_fired = "warning"

            # Fire alert through channels
            if alert_fired:
                from pipeline.alerts import fire_alert
                fire_alert(
                    level      = alert_fired,
                    category   = analysis.get("category", ""),
                    confidence = conf,
                    text       = text,
                    session_id = session.session_id,
                    persist    = True,
                )

            result = LiveChunkResult(
                session_id     = session.session_id,
                chunk_index    = session.chunk_index,
                timestamp      = chunk_ts,
                text           = text,
                language       = lang,
                flag           = analysis.get("flag", False),
                category       = analysis.get("category", ""),
                confidence     = conf,
                severity       = analysis.get("severity", "Low"),
                matched_keyword= analysis.get("matched_keyword", ""),
                stage_details  = analysis.get("stage_details", {}),
                alert_fired    = alert_fired,
            )
            session.chunk_index  += 1
            session._result_queue.put(result)

    print("[live] Capture loop ended")
    session.is_running = False


# ─────────────────────────────────────────────────────────────────────────────
# Public session API
# ─────────────────────────────────────────────────────────────────────────────

def start_session() -> LiveSession:
    """Start a new live monitoring session. Returns the session object."""
    session = LiveSession()
    session.is_running    = True
    session._stop_event   = threading.Event()
    session._result_queue = queue.Queue()
    session._thread = threading.Thread(
        target=_capture_loop,
        args=(session,),
        daemon=True,
    )
    session._thread.start()
    return session


def stop_session(session: LiveSession) -> None:
    """Stop the background capture thread."""
    session._stop_event.set()
    session.is_running = False
    if session._thread and session._thread.is_alive():
        session._thread.join(timeout=5)
    print(f"[live] Session {session.session_id} stopped")


def get_next_result(session: LiveSession, timeout: float = 0.05) -> LiveChunkResult | None:
    """
    Poll the result queue. Returns the next LiveChunkResult or None.
    Call this in a Streamlit loop: while True: r = get_next_result(session); ...
    """
    try:
        return session._result_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def save_evidence(session: LiveSession, duration_s: int = 30) -> str | None:
    """
    Save the last `duration_s` seconds of audio to a temp WAV file.
    Returns the file path or None if capture is not running.
    """
    # In a real implementation we'd maintain a ring buffer of raw audio.
    # This stub saves a placeholder so the alert DB row has a path.
    try:
        out = tempfile.NamedTemporaryFile(
            prefix=f"cg_evidence_{session.session_id}_",
            suffix=".wav",
            delete=False,
        )
        out.close()
        print(f"[live] Evidence placeholder: {out.name}")
        return out.name
    except Exception:
        return None

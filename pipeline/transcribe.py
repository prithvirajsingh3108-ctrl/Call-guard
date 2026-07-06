"""
transcribe.py
─────────────
Wraps whisperx to produce speaker-diarized, word-level timestamped
transcripts from a WAV audio file.

Step 1 mode  (USE_DIARIZATION=false in .env):
    Plain openai-whisper transcription — no speaker labels.
    Good for verifying your install before dealing with diarization.

Step 2 mode  (USE_DIARIZATION=true or by default):
    Full whisperx pipeline:
      1. Transcribe with whisperx (faster-whisper under the hood)
      2. Align word timestamps with phoneme-level models
      3. Diarize speakers with pyannote.audio
      4. Assign speaker labels to each segment

Output schema (list of dicts):
    [
      {
        "speaker":  "SPEAKER_00",
        "text":     "Hello, how are you?",
        "start":    0.0,   # seconds
        "end":      2.3
      },
      ...
    ]

Usage:
    # With diarization (default):
    python pipeline/transcribe.py sample_audio/test.wav

    # Plain whisper only (skip diarization):
    USE_DIARIZATION=false python pipeline/transcribe.py sample_audio/test.wav
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config from environment ───────────────────────────────────────────────────
WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "base")
COMPUTE_DEVICE   = os.getenv("COMPUTE_DEVICE", "cpu")
HF_TOKEN         = os.getenv("HF_TOKEN", "")
USE_DIARIZATION  = os.getenv("USE_DIARIZATION", "true").lower() != "false"

# ── Model cache — loaded once per process, reused across calls ────────────────
# Loading whisperx + diarization models takes 20-40s the first time.
# Caching them here means the second call is instant.
_whisperx_model   = None
_align_models     = {}   # keyed by language code
_diarize_model    = None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Plain Whisper transcription (no speaker labels)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_plain(audio_path: str) -> list[dict]:
    """
    Run plain openai-whisper transcription on a WAV file.

    Returns a list of segments with 'text', 'start', 'end' keys.
    Speaker is always 'UNKNOWN' because plain whisper has no diarization.
    """
    print(f"[transcribe] Loading plain Whisper model: {WHISPER_MODEL}")

    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    model = whisper.load_model(WHISPER_MODEL, device=COMPUTE_DEVICE)
    print(f"[transcribe] Transcribing: {audio_path}")

    # Detect language explicitly before transcribing
    forced_lang = os.getenv("WHISPER_LANGUAGE", "").strip() or None
    if forced_lang:
        detected_language = forced_lang
        lang_confidence   = 1.0
        print(f"[transcribe] Language forced: '{detected_language}'")
    else:
        import torch
        audio_for_detect = whisper.load_audio(audio_path)
        audio_for_detect = whisper.pad_or_trim(audio_for_detect)
        mel = whisper.log_mel_spectrogram(audio_for_detect).to(model.device)
        _, lang_probs     = model.detect_language(mel)
        detected_language = max(lang_probs, key=lang_probs.get)
        lang_confidence   = lang_probs[detected_language]

    print(f"[transcribe] ── Language Detection ──────────────────────────")
    print(f"[transcribe]   Detected language : {detected_language}")
    print(f"[transcribe]   Confidence        : {lang_confidence:.2%}")
    print(f"[transcribe]   Translation       : NO (task=transcribe)")
    print(f"[transcribe] ────────────────────────────────────────────────")

    result = model.transcribe(
        audio_path,
        language=detected_language,
        task="transcribe",
        verbose=False,
    )

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "speaker": "UNKNOWN",
            "text":    seg["text"].strip(),
            "start":   round(seg["start"], 3),
            "end":     round(seg["end"], 3),
        })

    print(f"[transcribe] Done — {len(segments)} segments")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — whisperx: transcription + alignment + diarization
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_with_diarization(audio_path: str) -> list[dict]:
    """
    Full whisperx pipeline:
      1. Transcribe
      2. Align word timestamps
      3. Diarize (requires HF_TOKEN)
      4. Merge speaker labels into segments

    Returns segments with 'speaker', 'text', 'start', 'end'.
    """
    # ── Validate token ────────────────────────────────────────────────────────
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set in .env\n"
            "Diarization requires a Hugging Face token.\n"
            "See .env.example for instructions, or set "
            "USE_DIARIZATION=false to skip diarization."
        )

    try:
        import whisperx
    except ImportError:
        raise RuntimeError(
            "whisperx is not installed. Run: pip install whisperx"
        )

    import warnings
    warnings.filterwarnings("ignore", message="torchcodec is not installed")

    # DiarizationPipeline moved to whisperx.diarize in whisperx >= 3.x
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    # ── Step 2a: Transcribe (use cached model if available) ───────────────────
    global _whisperx_model, _align_models, _diarize_model

    if _whisperx_model is None:
        print(f"[transcribe] Loading whisperx model: {WHISPER_MODEL} on {COMPUTE_DEVICE}")
        _whisperx_model = whisperx.load_model(
            WHISPER_MODEL,
            COMPUTE_DEVICE,
            compute_type="float32",
        )
    else:
        print(f"[transcribe] Using cached whisperx model")

    print(f"[transcribe] Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    print("[transcribe] Transcribing...")
    # Detect language explicitly with confidence — never rely on auto-detect
    forced_lang = os.getenv("WHISPER_LANGUAGE", "").strip() or None
    if forced_lang:
        detected_language = forced_lang
        lang_confidence   = 1.0
        print(f"[transcribe] Language forced via env: '{detected_language}'")
    else:
        try:
            detect_audio      = audio[:30 * 16000] if len(audio) > 30 * 16000 else audio
            # Use faster-whisper's detect_language which returns (code, confidence, probs)
            lang_code, lang_confidence, _ = _whisperx_model.model.detect_language(detect_audio)
            detected_language = lang_code
            if lang_confidence < 0.6 and len(audio) > 30 * 16000:
                lang_code2, lang_confidence2, _ = _whisperx_model.model.detect_language(audio)
                detected_language = lang_code2
                lang_confidence   = lang_confidence2
                print(f"[transcribe] Low confidence on 30s — retried on full audio")
        except Exception as exc:
            print(f"[transcribe] detect_language failed ({exc}), using transcribe fallback")
            tmp = _whisperx_model.transcribe(audio[:30*16000] if len(audio)>30*16000 else audio, batch_size=4)
            detected_language = tmp.get("language", "en")
            lang_confidence   = 0.0

    print(f"[transcribe] ── Language Detection ──────────────────────────")
    print(f"[transcribe]   Detected language : {detected_language}")
    print(f"[transcribe]   Confidence        : {lang_confidence:.2%}")
    print(f"[transcribe]   Translation       : NO (task=transcribe)")
    print(f"[transcribe] ────────────────────────────────────────────────")

    result = _whisperx_model.transcribe(
        audio,
        batch_size=16,
        language=detected_language,
        task="transcribe",
    )

    # ── Step 2b: Align word timestamps — SKIPPED ─────────────────────────────
    # Alignment downloads a 3-4GB language-specific model and is not needed
    # for threat detection (we only use segment-level text, not word timestamps).
    print("[transcribe] Skipping alignment (not needed for threat detection)")

    # ── Step 2c: Diarize speakers (cache diarization model) ───────────────────
    print("[transcribe] Running speaker diarization...")
    try:
        if _diarize_model is None:
            _diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                token=HF_TOKEN,
                device=COMPUTE_DEVICE,
            )
        else:
            print("[transcribe] Using cached diarization model")
        diarize_segments = _diarize_model(audio_path)

        # ── Step 2d: Assign speaker labels ────────────────────────────────────
        print("[transcribe] Assigning speaker labels...")
        result = assign_word_speakers(diarize_segments, result)

    except Exception as exc:
        print(
            f"[transcribe] Warning: diarization failed ({type(exc).__name__}: {exc})\n"
            f"[transcribe] Falling back to no-diarization mode.\n"
            f"  Accept licenses at:\n"
            f"  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            f"  https://huggingface.co/pyannote/segmentation-3.0"
        )

    # ── Build clean output list ───────────────────────────────────────────────
    # result may be a dict (from transcribe/align) or an AlignedTranscriptionResult object
    raw_segments = result.get("segments", []) if isinstance(result, dict) else result.segments
    segments = []
    for seg in raw_segments:
        # seg may be a dict or a dataclass depending on whisperx version
        if isinstance(seg, dict):
            speaker = seg.get("speaker", "UNKNOWN")
            text    = seg.get("text", "").strip()
            start   = round(seg.get("start", 0.0), 3)
            end     = round(seg.get("end", 0.0), 3)
        else:
            speaker = getattr(seg, "speaker", "UNKNOWN") or "UNKNOWN"
            text    = getattr(seg, "text", "").strip()
            start   = round(getattr(seg, "start", 0.0), 3)
            end     = round(getattr(seg, "end", 0.0), 3)

        segments.append({
            "speaker": speaker,
            "text":    text,
            "start":   start,
            "end":     end,
        })

    print(f"[transcribe] Done — {len(segments)} segments")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(audio_path: str) -> list[dict]:
    """
    Main entry point. Automatically pre-processes audio then transcribes.

    Uses diarization by default; set USE_DIARIZATION=false in .env to skip.

    Args:
        audio_path: Path to any supported audio file.

    Returns:
        List of segment dicts: [{speaker, text, start, end}, ...]
    """
    from pipeline.preprocess_audio import convert_to_wav

    # Convert to 16 kHz mono WAV first
    wav_path = convert_to_wav(audio_path)

    try:
        if USE_DIARIZATION:
            segments = transcribe_with_diarization(wav_path)
        else:
            segments = transcribe_plain(wav_path)
    finally:
        # Clean up the temp WAV if it was created (i.e., input wasn't already WAV)
        if wav_path != str(Path(audio_path).with_suffix(".wav")) and Path(wav_path).exists():
            # Only delete if it's a temp file (not next to the original)
            if not str(Path(audio_path).parent) in wav_path:
                os.unlink(wav_path)

    return segments


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/transcribe.py <audio_file>")
        print("       USE_DIARIZATION=false python pipeline/transcribe.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"CallGuard Transcriber")
    print(f"Mode: {'diarization' if USE_DIARIZATION else 'plain whisper'}")
    print(f"File: {audio_file}")
    print(f"{'='*60}\n")

    segments = transcribe(audio_file)

    print(f"\n{'='*60}")
    print(f"TRANSCRIPT ({len(segments)} segments)")
    print(f"{'='*60}")
    for seg in segments:
        print(f"[{seg['start']:6.2f}s – {seg['end']:6.2f}s]  {seg['speaker']}: {seg['text']}")

    # Also dump JSON for inspection
    output_json = Path(audio_file).stem + "_transcript.json"
    with open(output_json, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"\nTranscript saved to: {output_json}")

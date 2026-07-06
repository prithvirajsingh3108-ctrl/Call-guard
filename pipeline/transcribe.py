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
# Set USE_DIARIZATION=false to run Step-1 plain-whisper mode
USE_DIARIZATION  = os.getenv("USE_DIARIZATION", "true").lower() != "false"


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

    result = model.transcribe(audio_path, verbose=False)

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

    # Suppress the noisy torchcodec warning on macOS — it's harmless here
    # because we pass pre-loaded numpy audio directly to whisperx
    import warnings
    warnings.filterwarnings("ignore", message="torchcodec is not installed")

    # DiarizationPipeline moved to whisperx.diarize in whisperx >= 3.x
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    # ── Step 2a: Transcribe ───────────────────────────────────────────────────
    print(f"[transcribe] Loading whisperx model: {WHISPER_MODEL} on {COMPUTE_DEVICE}")
    model = whisperx.load_model(
        WHISPER_MODEL,
        COMPUTE_DEVICE,
        compute_type="float32",   # use float16 for CUDA speedup
    )

    print(f"[transcribe] Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    print("[transcribe] Transcribing...")
    result = model.transcribe(audio, batch_size=16)

    detected_language = result.get("language", "en")
    print(f"[transcribe] Detected language: {detected_language}")

    # ── Step 2b: Align word timestamps ────────────────────────────────────────
    print("[transcribe] Aligning word timestamps...")
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=COMPUTE_DEVICE,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            COMPUTE_DEVICE,
            return_char_alignments=False,
        )
    except Exception as exc:
        # Alignment can fail for some languages — carry on without it
        print(f"[transcribe] Warning: alignment failed ({exc}), continuing without it.")

    # ── Step 2c: Diarize speakers ─────────────────────────────────────────────
    # Pass the WAV file path directly — pyannote reads it via soundfile,
    # which avoids the torchcodec/ffmpeg dylib incompatibility on macOS
    print("[transcribe] Running speaker diarization...")
    try:
        diarize_model = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1",
            token=HF_TOKEN,
            device=COMPUTE_DEVICE,
        )
        diarize_segments = diarize_model(audio_path)

        # ── Step 2d: Assign speaker labels to transcript segments ─────────────
        print("[transcribe] Assigning speaker labels...")
        result = assign_word_speakers(diarize_segments, result)

    except Exception as exc:
        # Diarization failed (e.g. gated model not yet approved on HuggingFace).
        # Fall back gracefully — transcript still works, speakers labeled UNKNOWN.
        print(
            f"[transcribe] Warning: diarization failed ({type(exc).__name__}: {exc})\n"
            f"[transcribe] Falling back to no-diarization mode. "
            f"To enable speaker labels, accept the model license at:\n"
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

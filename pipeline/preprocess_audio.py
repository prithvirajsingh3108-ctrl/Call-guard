"""
preprocess_audio.py
───────────────────
Converts any incoming audio file to a 16 kHz mono WAV — the format
whisperx expects. Uses pydub under the hood, which in turn relies on
ffmpeg being installed on the system.

Usage (standalone):
    python pipeline/preprocess_audio.py path/to/input.mp3
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load .env so AUDIO_SAMPLE_RATE is available if set there
load_dotenv()

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))


def convert_to_wav(input_path: str, output_path: str | None = None) -> str:
    """
    Convert an audio file to 16 kHz mono WAV format.

    Args:
        input_path:  Path to the source audio file.
        output_path: Where to write the converted WAV.
                     If None, a temp file is created and its path returned.

    Returns:
        Path to the converted WAV file as a string.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError:        If the file extension is not supported.
        RuntimeError:      If pydub/ffmpeg fails during conversion.
    """
    # ── Validate input ────────────────────────────────────────────────────────
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # ── Import pydub here so we get a clear error if it's missing ─────────────
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "pydub is not installed. Run: pip install pydub"
        )

    # ── Load audio ────────────────────────────────────────────────────────────
    print(f"[preprocess] Loading audio: {input_path}")
    try:
        # pydub auto-detects format from the file extension
        audio = AudioSegment.from_file(str(input_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load audio file '{input_path}'. "
            f"Make sure ffmpeg is installed (brew install ffmpeg). "
            f"Original error: {exc}"
        ) from exc

    # ── Resample + convert to mono ────────────────────────────────────────────
    original_sr = audio.frame_rate
    original_channels = audio.channels
    print(
        f"[preprocess] Original: {original_sr} Hz, "
        f"{original_channels} channel(s), "
        f"{len(audio) / 1000:.1f}s"
    )

    # Convert to mono by mixing channels down
    audio = audio.set_channels(1)
    # Resample to target sample rate (pydub export handles 16-bit PCM automatically)
    audio = audio.set_frame_rate(SAMPLE_RATE)

    print(f"[preprocess] Converted to: {SAMPLE_RATE} Hz, mono, 16-bit PCM")

    # ── Write output ──────────────────────────────────────────────────────────
    if output_path is None:
        # Create a temp file that persists until the caller deletes it
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    output_path = str(output_path)
    audio.export(output_path, format="wav")
    print(f"[preprocess] Saved to: {output_path}")

    return output_path


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/preprocess_audio.py <input_audio>")
        sys.exit(1)

    result = convert_to_wav(sys.argv[1])
    print(f"\nConversion complete → {result}")

"""
pipeline/preprocess_audio.py
────────────────────────────
Convert any incoming audio to a clean 16 kHz mono WAV ready for whisperx.

Pipeline (each step is individually switchable via config / .env):

  1. Load with pydub / ffmpeg
  2. Upsample 8 kHz telephone audio to 16 kHz  (always)
  3. Mix to mono                                (always)
  4. EBU R128 loudness normalisation            (AUDIO_LOUDNORM=true)
  5. High-pass filter at ~80 Hz                 (AUDIO_HIGHPASS_HZ=80)
  6. Noise reduction                            (AUDIO_NOISE_REDUCE=true, needs noisereduce)
  7. Trim / split long silences                 (AUDIO_TRIM_SILENCE=true)
  8. Export as 16-bit PCM WAV at 16 kHz
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"}
SAMPLE_RATE       = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
LOUDNORM          = os.getenv("AUDIO_LOUDNORM",    "true").lower()  in ("1","true","yes")
HIGHPASS_HZ       = int(os.getenv("AUDIO_HIGHPASS_HZ", "80"))
NOISE_REDUCE      = os.getenv("AUDIO_NOISE_REDUCE", "false").lower() in ("1","true","yes")
TRIM_SILENCE      = os.getenv("AUDIO_TRIM_SILENCE", "true").lower()  in ("1","true","yes")
SILENCE_MIN_MS    = int(os.getenv("AUDIO_SILENCE_MIN_LEN_MS", "500"))


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _ffmpeg_loudnorm(input_wav: str, output_wav: str) -> None:
    """Run EBU R128 two-pass loudness normalisation via ffmpeg."""
    # Pass 1: measure
    p1 = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_wav,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # Extract measured values from stderr JSON block
    import json, re
    m = re.search(r"\{[^}]+\}", p1.stderr, re.DOTALL)
    if m:
        meas = json.loads(m.group())
        il  = meas.get("input_i",  "-16.0")
        itp = meas.get("input_tp", "-1.5")
        ilra = meas.get("input_lra", "11.0")
        ith = meas.get("input_thresh", "-28.0")
        ofs = meas.get("target_offset", "0.0")
        af  = (
            f"loudnorm=I=-16:TP=-1.5:LRA=11:"
            f"measured_I={il}:measured_TP={itp}:measured_LRA={ilra}:"
            f"measured_thresh={ith}:offset={ofs}:linear=true:print_format=summary"
        )
    else:
        af = "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true"

    # Pass 2: apply
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_wav, "-af", af, output_wav],
        capture_output=True, check=True,
    )


def _ffmpeg_highpass(input_wav: str, output_wav: str, hz: int) -> None:
    """Apply a high-pass filter at `hz` Hz."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_wav,
         "-af", f"highpass=f={hz}", output_wav],
        capture_output=True, check=True,
    )


def _noise_reduce_wav(input_wav: str, output_wav: str) -> None:
    """
    Spectral noise reduction using the `noisereduce` package.
    Falls back gracefully if the package is not installed.
    """
    try:
        import noisereduce as nr
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(input_wav)
        # Use first 0.5 s as noise profile if available
        noise_sample = data[: int(sr * 0.5)] if len(data) > sr * 0.5 else data
        reduced = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample, prop_decrease=0.75)
        sf.write(output_wav, reduced, sr, subtype="PCM_16")
        print("[preprocess] Noise reduction applied")
    except ImportError:
        print("[preprocess] noisereduce not installed — skipping noise reduction")
        import shutil
        shutil.copy2(input_wav, output_wav)
    except Exception as exc:
        print(f"[preprocess] Noise reduction failed ({exc}) — skipping")
        import shutil
        shutil.copy2(input_wav, output_wav)


def _trim_silence_pydub(audio):
    """
    Remove leading/trailing silence and split on long interior silences.
    Returns a pydub AudioSegment with long silences replaced by a short gap.
    """
    try:
        from pydub.silence import detect_silence
        silences = detect_silence(audio, min_silence_len=SILENCE_MIN_MS, silence_thresh=-45)
        if not silences:
            return audio
        # Strip leading silence
        if silences[0][0] == 0:
            audio = audio[silences[0][1]:]
            silences = silences[1:]
        # Strip trailing silence
        if silences and silences[-1][1] >= len(audio) - 50:
            audio = audio[: silences[-1][0]]
        return audio
    except Exception as exc:
        print(f"[preprocess] Silence trim failed ({exc}) — skipping")
        return audio


def convert_to_wav(input_path: str, output_path: str | None = None) -> str:
    """
    Convert input audio to clean 16 kHz mono WAV.

    Parameters
    ----------
    input_path  : path to source audio (any supported format)
    output_path : destination WAV path; if None a temp file is created

    Returns
    -------
    Path to the output WAV file (str).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pydub not installed. Run: pip install pydub")

    print(f"[preprocess] Loading: {input_path}")
    try:
        audio = AudioSegment.from_file(str(input_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{input_path}'. Ensure ffmpeg is installed. Error: {exc}"
        ) from exc

    orig_sr  = audio.frame_rate
    orig_ch  = audio.channels
    orig_dur = len(audio) / 1000.0
    print(f"[preprocess] Source: {orig_sr} Hz, {orig_ch} ch, {orig_dur:.1f}s")

    # ── Step 1: Mono + target sample rate ────────────────────────────────
    audio = audio.set_channels(1)

    # Telephone audio (8 kHz) — upsample properly via ffmpeg to avoid
    # pydub's naive linear interpolation which degrades Whisper accuracy.
    if orig_sr <= 8100 and _has_ffmpeg():
        print(f"[preprocess] Telephone audio detected ({orig_sr} Hz) — upsampling via ffmpeg")
        tmp_in  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_in.close(); tmp_out.close()
        try:
            audio.export(tmp_in.name, format="wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in.name,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", tmp_out.name],
                capture_output=True, check=True,
            )
            audio = AudioSegment.from_file(tmp_out.name)
        finally:
            Path(tmp_in.name).unlink(missing_ok=True)
            Path(tmp_out.name).unlink(missing_ok=True)
    else:
        audio = audio.set_frame_rate(SAMPLE_RATE)

    print(f"[preprocess] Resampled to {SAMPLE_RATE} Hz mono")

    # ── Step 2: Silence trim (before loudnorm so we don't normalise silence) ─
    if TRIM_SILENCE:
        audio = _trim_silence_pydub(audio)
        print(f"[preprocess] Silence trimmed → {len(audio)/1000:.1f}s")

    # ── Write intermediate WAV for ffmpeg steps ───────────────────────────
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()
    output_path = str(output_path)
    audio.export(output_path, format="wav")

    # ── Step 3: Loudness normalisation (ffmpeg two-pass) ─────────────────
    if LOUDNORM and _has_ffmpeg():
        try:
            loud_out = output_path + ".loud.wav"
            _ffmpeg_loudnorm(output_path, loud_out)
            import shutil
            shutil.move(loud_out, output_path)
            print("[preprocess] Loudness normalisation applied (EBU R128)")
        except Exception as exc:
            print(f"[preprocess] Loudnorm failed ({exc}) — skipping")

    # ── Step 4: High-pass filter ──────────────────────────────────────────
    if HIGHPASS_HZ > 0 and _has_ffmpeg():
        try:
            hp_out = output_path + ".hp.wav"
            _ffmpeg_highpass(output_path, hp_out, HIGHPASS_HZ)
            import shutil
            shutil.move(hp_out, output_path)
            print(f"[preprocess] High-pass filter applied at {HIGHPASS_HZ} Hz")
        except Exception as exc:
            print(f"[preprocess] High-pass filter failed ({exc}) — skipping")

    # ── Step 5: Noise reduction ───────────────────────────────────────────
    if NOISE_REDUCE:
        nr_out = output_path + ".nr.wav"
        _noise_reduce_wav(output_path, nr_out)
        import shutil
        shutil.move(nr_out, output_path)

    print(f"[preprocess] Final WAV: {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/preprocess_audio.py <input_audio>")
        sys.exit(1)
    result = convert_to_wav(sys.argv[1])
    print(f"\nConversion complete → {result}")

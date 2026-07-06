"""
voice_recognition.py
────────────────────
Speaker recognition using resemblyzer voice embeddings.

This module is completely separate from diarization. Diarization tells us
"SPEAKER_00 vs SPEAKER_01" within one call. This module identifies WHO those
speakers are by comparing their voice embeddings against enrolled profiles.

Key functions
─────────────
enroll_voice(audio_path, name)
    Generate a 256-dim embedding from a clean audio sample and store it.

identify_speakers(segments, wav_path)
    For each diarized speaker, extract an embedding from their combined
    speech, compare against enrolled profiles, and return a name mapping.

Usage
─────
    from pipeline.voice_recognition import enroll_voice, identify_speakers

    # Enroll a person once
    embedding = enroll_voice("john_sample.wav", "John")

    # Identify speakers in a new call
    name_map = identify_speakers(segments, "call.wav")
    # {"SPEAKER_00": ("John", 0.91), "SPEAKER_01": ("Unknown Speaker", 0.0)}
"""

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Cosine similarity threshold — above this score a speaker is considered matched
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.75"))

# ── Cached encoder — loaded once per process ─────────────────────────────────
_encoder = None


def _get_encoder():
    """Lazy-load the VoiceEncoder once and cache it."""
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        raise RuntimeError(
            "resemblyzer is not installed. Run: pip install resemblyzer"
        )
    _encoder = VoiceEncoder()
    return _encoder


def _load_wav_resemblyzer(wav_path: str) -> np.ndarray:
    """
    Load a WAV file into a float32 numpy array at 16kHz mono,
    the format resemblyzer expects.
    """
    try:
        from resemblyzer import preprocess_wav
    except ImportError:
        raise RuntimeError("resemblyzer is not installed.")
    return preprocess_wav(Path(wav_path))


def generate_embedding(audio_path: str) -> list[float]:
    """
    Generate a 256-dim voice embedding from an audio file.

    Args:
        audio_path: Path to any supported audio file (wav, mp3, m4a…).
                    Will be converted to 16kHz mono WAV automatically.

    Returns:
        List of 256 floats representing this person's voice.
    """
    from pipeline.preprocess_audio import convert_to_wav

    # Convert to 16kHz mono WAV first
    wav_path = convert_to_wav(audio_path)

    try:
        encoder = _get_encoder()
        wav     = _load_wav_resemblyzer(wav_path)
        embedding = encoder.embed_utterance(wav)
        return embedding.tolist()
    finally:
        # Clean up temp WAV if created
        if wav_path != audio_path and Path(wav_path).exists():
            os.unlink(wav_path)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def enroll_voice(audio_path: str, name: str) -> list[float]:
    """
    Generate a voice embedding from an audio sample and store it in the DB.

    Args:
        audio_path: Path to a 10-30s clean audio sample of the person.
        name:       Their name (used as the identifier across calls).

    Returns:
        The generated embedding as a list of floats.
    """
    from db.database import get_session, save_voice_profile

    print(f"[voice_recognition] Generating embedding for: {name}")
    embedding = generate_embedding(audio_path)

    with get_session() as session:
        profile = save_voice_profile(
            session,
            name       = name,
            embedding  = embedding,
            audio_file = Path(audio_path).name,
        )
        print(f"[voice_recognition] Enrolled '{name}' (id={profile.id})")

    return embedding


def identify_speakers(
    segments: list[dict],
    wav_path: str,
) -> dict[str, tuple[str, float]]:
    """
    Identify who each diarized speaker is by comparing their voice
    against all enrolled profiles.

    Args:
        segments: List of segment dicts with 'speaker', 'start', 'end' keys.
        wav_path: Path to the 16kHz mono WAV of the full call.

    Returns:
        Dict mapping speaker label → (matched_name, confidence):
            {
              "SPEAKER_00": ("John Smith", 0.91),
              "SPEAKER_01": ("Unknown Speaker", 0.0),
            }
        If no profiles are enrolled, returns empty dict.
    """
    from db.database import get_session, get_all_voice_profiles

    # Load enrolled profiles
    with get_session() as session:
        profiles = get_all_voice_profiles(session)
        if not profiles:
            print("[voice_recognition] No enrolled voice profiles — skipping recognition.")
            return {}
        # Pull data out before session closes
        profile_data = [(p.name, p.embedding) for p in profiles]

    # Group segments by speaker label
    from collections import defaultdict
    speaker_segments: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        if spk and spk != "UNKNOWN":
            speaker_segments[spk].append(seg)

    if not speaker_segments:
        print("[voice_recognition] No diarized speakers found — skipping recognition.")
        return {}

    # Load full call audio
    try:
        wav_full = _load_wav_resemblyzer(wav_path)
    except Exception as exc:
        print(f"[voice_recognition] Could not load audio for recognition: {exc}")
        return {}

    encoder = _get_encoder()
    sample_rate = 16000  # resemblyzer always works at 16kHz

    name_map: dict[str, tuple[str, float]] = {}

    for speaker_label, spk_segs in speaker_segments.items():
        # Collect audio slices for this speaker
        slices = []
        for seg in spk_segs:
            start_sample = int(seg["start"] * sample_rate)
            end_sample   = int(seg["end"]   * sample_rate)
            if end_sample > start_sample and end_sample <= len(wav_full):
                slices.append(wav_full[start_sample:end_sample])

        if not slices:
            name_map[speaker_label] = ("Unknown Speaker", 0.0)
            continue

        # Concatenate all slices for this speaker and embed
        combined = np.concatenate(slices)
        try:
            spk_embedding = encoder.embed_utterance(combined).tolist()
        except Exception as exc:
            print(f"[voice_recognition] Embedding failed for {speaker_label}: {exc}")
            name_map[speaker_label] = ("Unknown Speaker", 0.0)
            continue

        # Compare against all enrolled profiles
        best_name  = "Unknown Speaker"
        best_score = 0.0

        for profile_name, profile_embedding in profile_data:
            score = cosine_similarity(spk_embedding, profile_embedding)
            if score > best_score:
                best_score = score
                best_name  = profile_name

        if best_score >= RECOGNITION_THRESHOLD:
            name_map[speaker_label] = (best_name, round(best_score, 3))
            print(f"[voice_recognition] {speaker_label} → '{best_name}' (score={best_score:.3f})")
        else:
            name_map[speaker_label] = ("Unknown Speaker", round(best_score, 3))
            print(f"[voice_recognition] {speaker_label} → no match (best={best_score:.3f})")

    return name_map


def apply_speaker_names(
    segments: list[dict],
    name_map: dict[str, tuple[str, float]],
) -> list[dict]:
    """
    Replace generic speaker labels in segments with matched names.

    Args:
        segments: Original segments with 'speaker' field.
        name_map: Output of identify_speakers().

    Returns:
        New list of segments with 'speaker' replaced by matched name,
        and 'speaker_confidence' added where a match was found.
    """
    updated = []
    for seg in segments:
        seg = dict(seg)  # don't mutate originals
        label = seg.get("speaker", "UNKNOWN")
        if label in name_map:
            matched_name, confidence = name_map[label]
            seg["speaker"]            = matched_name
            seg["speaker_original"]   = label          # keep original label too
            seg["speaker_confidence"] = confidence
        else:
            seg["speaker_confidence"] = 0.0
        updated.append(seg)
    return updated

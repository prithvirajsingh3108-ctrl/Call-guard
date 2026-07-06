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
) -> dict[str, tuple[str, float, str]]:
    """
    Identify who each diarized speaker is by comparing their voice
    against all enrolled profiles in the database.

    Handles all edge cases:
    - No enrolled profiles → returns mapping with status "no_enrolled_voices"
    - Speaker label is UNKNOWN (diarization off) → still attempts matching
    - Audio slice too short → skips that speaker gracefully

    Args:
        segments: List of segment dicts with 'speaker', 'start', 'end' keys.
        wav_path: Path to the 16kHz mono WAV of the full call.

    Returns:
        Dict mapping speaker label → (matched_name, confidence, status):
            {
              "SPEAKER_00": ("John Smith", 0.91, "matched"),
              "SPEAKER_01": ("Unknown Speaker", 0.43, "no_match"),
              "UNKNOWN":    ("Unknown Speaker", 0.0,  "no_enrolled_voices"),
            }
    """
    from db.database import get_session, get_all_voice_profiles
    from collections import defaultdict

    THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.75"))

    # ── Load enrolled profiles ────────────────────────────────────────────────
    with get_session() as session:
        profiles = get_all_voice_profiles(session)
        if not profiles:
            print("[voice_recognition] No enrolled voice profiles in database.")
            # Return a mapping for every unique speaker label — all no_enrolled_voices
            unique_speakers = {seg.get("speaker", "UNKNOWN") for seg in segments}
            return {
                spk: ("Unknown Speaker (no enrolled voices yet)", 0.0, "no_enrolled_voices")
                for spk in unique_speakers
            }
        profile_data = [(p.name, p.embedding) for p in profiles]
        print(f"[voice_recognition] Loaded {len(profile_data)} enrolled profile(s): "
              f"{[p[0] for p in profile_data]}")

    # ── Group segments by speaker label ──────────────────────────────────────
    speaker_segments: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN") or "UNKNOWN"
        speaker_segments[spk].append(seg)

    if not speaker_segments:
        print("[voice_recognition] No segments found.")
        return {}

    # ── Load full call WAV ────────────────────────────────────────────────────
    try:
        wav_full = _load_wav_resemblyzer(wav_path)
    except Exception as exc:
        print(f"[voice_recognition] Could not load audio: {exc}")
        return {
            spk: ("Unknown Speaker", 0.0, "no_match")
            for spk in speaker_segments
        }

    encoder     = _get_encoder()
    sample_rate = 16000
    name_map: dict[str, tuple[str, float, str]] = {}

    # ── For each unique speaker, extract embedding and compare ────────────────
    for speaker_label, spk_segs in speaker_segments.items():
        print(f"\n[voice_recognition] Processing speaker: '{speaker_label}' "
              f"({len(spk_segs)} segment(s))")

        # Collect audio slices for this speaker
        slices = []
        for seg in spk_segs:
            start_sample = int(seg["start"] * sample_rate)
            end_sample   = int(seg["end"]   * sample_rate)
            if end_sample > start_sample and end_sample <= len(wav_full):
                slices.append(wav_full[start_sample:end_sample])

        if not slices:
            print(f"[voice_recognition]   No valid audio slices — marking as Unknown Speaker")
            name_map[speaker_label] = ("Unknown Speaker", 0.0, "no_match")
            continue

        # Concatenate all slices and embed
        combined = np.concatenate(slices)
        print(f"[voice_recognition]   Combined audio: {len(combined)/sample_rate:.1f}s")

        try:
            spk_embedding = encoder.embed_utterance(combined).tolist()
        except Exception as exc:
            print(f"[voice_recognition]   Embedding failed: {exc}")
            name_map[speaker_label] = ("Unknown Speaker", 0.0, "no_match")
            continue

        # ── Compare against every enrolled profile — debug print each score ──
        best_name  = "Unknown Speaker"
        best_score = 0.0
        print(f"[voice_recognition]   Similarity scores against enrolled profiles:")
        for profile_name, profile_embedding in profile_data:
            score = cosine_similarity(spk_embedding, profile_embedding)
            threshold_marker = "✅ MATCH" if score >= THRESHOLD else "❌ below threshold"
            print(f"[voice_recognition]     vs '{profile_name}': {score:.4f} — {threshold_marker} (threshold={THRESHOLD})")
            if score > best_score:
                best_score = score
                best_name  = profile_name

        if best_score >= THRESHOLD:
            name_map[speaker_label] = (best_name, round(best_score, 3), "matched")
            print(f"[voice_recognition]   → MATCHED: '{best_name}' ({best_score:.3f})")
        else:
            name_map[speaker_label] = ("Unknown Speaker", round(best_score, 3), "no_match")
            print(f"[voice_recognition]   → NO MATCH (best score {best_score:.3f} < {THRESHOLD})")

    return name_map


def apply_speaker_names(
    segments: list[dict],
    name_map: dict[str, tuple[str, float, str]],
) -> list[dict]:
    """
    Enrich each segment with voice match results.

    Adds to each segment dict:
        matched_name:       resolved display name
        match_confidence:   similarity score 0.0–1.0
        match_status:       "matched" | "no_match" | "no_enrolled_voices" | "not_run"
        speaker_original:   original diarized label (e.g. "SPEAKER_00")
        speaker_display:    formatted label for UI (e.g. "John Smith (91% match)")

    The 'speaker' field is kept as the original diarized label for DB storage.
    Use 'speaker_display' for showing in the UI.
    """
    updated = []
    for seg in segments:
        seg = dict(seg)
        original_label = seg.get("speaker", "UNKNOWN") or "UNKNOWN"

        if original_label in name_map:
            matched_name, confidence, status = name_map[original_label]
        else:
            matched_name, confidence, status = "Unknown Speaker", 0.0, "not_run"

        seg["speaker_original"] = original_label
        seg["matched_name"]     = matched_name
        seg["match_confidence"] = confidence
        seg["match_status"]     = status

        # Build display label for UI
        if status == "matched":
            seg["speaker_display"] = f"{matched_name} ({confidence:.0%} match)"
        elif status == "no_enrolled_voices":
            seg["speaker_display"] = "Unknown Speaker (no enrolled voices yet)"
        elif status == "no_match":
            seg["speaker_display"] = f"Unknown Speaker (no match, best: {confidence:.0%})"
        else:
            seg["speaker_display"] = original_label

        updated.append(seg)
    return updated

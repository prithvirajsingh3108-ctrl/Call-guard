"""
detector.py
───────────
Two-pass threat detection over transcript segments.

PASS 1 — Fuzzy keyword matching
    Each segment's text is compared against every keyword in keywords.json
    using rapidfuzz partial_ratio. A match above FUZZY_THRESHOLD is a
    candidate flag.

PASS 2 — Context window analysis
    The segment is passed to `analyze_segment()` together with the N
    most-recent prior segments (the "context window"). This function is
    the SWAPPABLE interface — right now it's keyword-based, but you can
    drop in a classifier or external API call without changing anything
    outside this function.

    Interface contract for analyze_segment():
        Input:
            segment       — the current dict {speaker, text, start, end}
            context_window — list of preceding segment dicts (may be empty)
        Output:
            {
              "flag":       bool,   # True = flagged
              "category":   str,    # matched category or ""
              "confidence": float,  # 0.0 – 1.0
              "matched_keyword": str   # which keyword triggered it
            }

Usage (standalone):
    python pipeline/detector.py                   # uses built-in sample
    python pipeline/detector.py transcript.json   # pass a transcript file
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
KEYWORDS_PATH      = os.getenv("KEYWORDS_PATH", "pipeline/keywords.json")
FUZZY_THRESHOLD    = int(os.getenv("FUZZY_THRESHOLD", "80"))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "4"))


# ── Load keyword list ─────────────────────────────────────────────────────────

def load_keywords(path: str = KEYWORDS_PATH) -> dict[str, list[str]]:
    """
    Load and return the keyword dictionary from the JSON file.

    Returns:
        { "threat": [...], "abuse": [...], ... }
    """
    kw_path = Path(path)
    if not kw_path.exists():
        raise FileNotFoundError(
            f"Keywords file not found: {kw_path}\n"
            f"Expected at: {kw_path.resolve()}"
        )
    with open(kw_path, "r") as f:
        data = json.load(f)

    # Strip out meta keys that start with '_'
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ── Pass 1: Fuzzy keyword matcher ─────────────────────────────────────────────

def fuzzy_match(text: str, keywords: dict[str, list[str]]) -> list[dict]:
    """
    Compare `text` against all keywords using rapidfuzz partial matching.

    Returns a list of all matches above FUZZY_THRESHOLD, sorted by score:
        [{"category": "threat", "keyword": "...", "score": 95.0}, ...]
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        raise RuntimeError("rapidfuzz is not installed. Run: pip install rapidfuzz")

    text_lower = text.lower()
    matches = []

    for category, kw_list in keywords.items():
        for keyword in kw_list:
            # partial_ratio: checks if keyword appears as a substring-like match
            # This tolerates small insertions/deletions (misspellings, filler words)
            score = fuzz.partial_ratio(keyword.lower(), text_lower)
            if score >= FUZZY_THRESHOLD:
                matches.append({
                    "category": category,
                    "keyword":  keyword,
                    "score":    score,
                })

    # Highest scoring match first
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


# ── Pass 2: Context-aware analysis (SWAPPABLE) ────────────────────────────────

def analyze_segment(
    segment: dict,
    context_window: list[dict],
) -> dict:
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║  SWAPPABLE INTERFACE                                             ║
    ║  Replace the body of this function with a classifier, LLM call, ║
    ║  or external moderation API without touching anything else.      ║
    ╚══════════════════════════════════════════════════════════════════╝

    Determines whether `segment` should be flagged as threatening/harmful,
    taking into account the surrounding conversation context.

    Args:
        segment:        Current segment dict {speaker, text, start, end}.
        context_window: Up to CONTEXT_WINDOW_SIZE preceding segments.

    Returns:
        {
          "flag":            bool,
          "category":        str,   # e.g. "threat", "abuse", "" if not flagged
          "confidence":      float, # 0.0 – 1.0
          "matched_keyword": str,   # the specific keyword that triggered it
        }
    """
    # ── Current implementation: keyword-based with context dampening ──────────

    keywords = load_keywords()

    # First pass: does the segment itself match anything?
    segment_matches = fuzzy_match(segment["text"], keywords)

    if not segment_matches:
        # No match in this segment — not flagged
        return {
            "flag":            False,
            "category":        "",
            "confidence":      0.0,
            "matched_keyword": "",
        }

    best_match = segment_matches[0]
    raw_score  = best_match["score"] / 100.0   # normalise to 0-1

    # ── Context check: look for benign-signal phrases in context window ───────
    # If the surrounding conversation contains clear sarcasm/sports/work phrases,
    # dampen the confidence score. This is a simple heuristic — replace with
    # a real classifier for better accuracy.
    BENIGN_SIGNALS = [
        "at work", "killing it", "crushed it", "nailed it",
        "soccer", "football", "basketball", "game", "match",
        "movie", "film", "book", "chapter",
        "joke", "kidding", "just saying", "not serious",
    ]

    context_text = " ".join(s["text"].lower() for s in context_window)
    benign_hit = any(signal in context_text for signal in BENIGN_SIGNALS)

    if benign_hit:
        # Reduce confidence by 30% if benign signals are present nearby
        confidence = max(0.0, raw_score - 0.30)
    else:
        confidence = raw_score

    # Only flag if confidence clears a minimum bar after context dampening
    MIN_CONFIDENCE = 0.50
    flagged = confidence >= MIN_CONFIDENCE

    return {
        "flag":            flagged,
        "category":        best_match["category"] if flagged else "",
        "confidence":      round(confidence, 3),
        "matched_keyword": best_match["keyword"] if flagged else "",
    }


# ── Main pipeline: run detection over a full transcript ───────────────────────

def detect_threats(segments: list[dict]) -> list[dict]:
    """
    Run both passes over a full list of transcript segments.

    Returns a new list where each segment dict is enriched with
    detection results:
        {
          ...original segment fields...,
          "flag":            bool,
          "category":        str,
          "confidence":      float,
          "matched_keyword": str,
          "context_window":  [list of prior segment texts],
        }
    """
    keywords = load_keywords()
    results  = []

    for i, segment in enumerate(segments):
        # Build context window from the N segments before this one
        window_start   = max(0, i - CONTEXT_WINDOW_SIZE)
        context_window = segments[window_start:i]   # excludes current segment

        # Pass 1: quick fuzzy pre-filter (skip analyze_segment if no match)
        pass1_matches = fuzzy_match(segment["text"], keywords)

        if pass1_matches:
            # Pass 2: context-aware analysis
            analysis = analyze_segment(segment, context_window)
        else:
            # No keyword match at all — skip expensive pass 2
            analysis = {
                "flag":            False,
                "category":        "",
                "confidence":      0.0,
                "matched_keyword": "",
            }

        # Merge everything into one enriched dict
        enriched = {
            **segment,
            **analysis,
            "context_window": [
                {"speaker": s["speaker"], "text": s["text"]}
                for s in context_window
            ],
        }
        results.append(enriched)

    return results


def summarize_flags(results: list[dict]) -> dict:
    """
    Produce a call-level summary of detected flags.

    Returns:
        {
          "total_segments":  int,
          "total_flags":     int,
          "by_category":     {"threat": 2, "abuse": 1, ...},
          "highest_confidence": float,
          "flagged_segments": [subset of enriched results that were flagged],
        }
    """
    flagged = [r for r in results if r["flag"]]
    by_category: dict[str, int] = {}
    for r in flagged:
        cat = r["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "total_segments":     len(results),
        "total_flags":        len(flagged),
        "by_category":        by_category,
        "highest_confidence": max((r["confidence"] for r in flagged), default=0.0),
        "flagged_segments":   flagged,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample transcript for testing without a real audio file
    SAMPLE_TRANSCRIPT = [
        {"speaker": "SPEAKER_00", "text": "Hey, how's the project going?", "start": 0.0, "end": 2.5},
        {"speaker": "SPEAKER_01", "text": "We're totally killing it at work this week!", "start": 2.6, "end": 5.1},
        {"speaker": "SPEAKER_00", "text": "Nice. Hey, I need to talk to you about John.", "start": 5.2, "end": 8.0},
        {"speaker": "SPEAKER_01", "text": "What about him?", "start": 8.1, "end": 9.0},
        {"speaker": "SPEAKER_00", "text": "We need to get rid of him. He's been leaking our plans.", "start": 9.1, "end": 13.0},
        {"speaker": "SPEAKER_01", "text": "I agree. Nobody will find out if we handle this quietly.", "start": 13.1, "end": 17.0},
        {"speaker": "SPEAKER_00", "text": "I will kill you if you tell anyone about this.", "start": 17.1, "end": 20.5},
        {"speaker": "SPEAKER_01", "text": "Relax, I'm just kidding around.", "start": 20.6, "end": 22.0},
        {"speaker": "SPEAKER_00", "text": "Let's blow up this presentation tomorrow.", "start": 22.1, "end": 25.0},
    ]

    # Allow passing a JSON transcript file as argument
    if len(sys.argv) > 1:
        transcript_file = sys.argv[1]
        print(f"Loading transcript: {transcript_file}")
        with open(transcript_file) as f:
            transcript = json.load(f)
    else:
        print("No transcript file provided — using built-in sample transcript.")
        transcript = SAMPLE_TRANSCRIPT

    print(f"\n{'='*60}")
    print("CallGuard Threat Detector")
    print(f"Keywords file: {KEYWORDS_PATH}")
    print(f"Fuzzy threshold: {FUZZY_THRESHOLD}")
    print(f"Context window:  {CONTEXT_WINDOW_SIZE} segments")
    print(f"{'='*60}\n")

    results = detect_threats(transcript)
    summary = summarize_flags(results)

    # Print all segments, highlighting flagged ones
    for r in results:
        flag_marker = " ⚠ FLAGGED" if r["flag"] else ""
        print(
            f"[{r['start']:6.2f}s]  {r['speaker']}: {r['text']}{flag_marker}"
        )
        if r["flag"]:
            print(
                f"          → category={r['category']}  "
                f"confidence={r['confidence']:.2f}  "
                f"keyword='{r['matched_keyword']}'"
            )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total segments : {summary['total_segments']}")
    print(f"Total flags    : {summary['total_flags']}")
    print(f"By category    : {summary['by_category']}")
    print(f"Max confidence : {summary['highest_confidence']:.2f}")

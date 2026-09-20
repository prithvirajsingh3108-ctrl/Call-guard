"""
pipeline/detector.py
────────────────────
Multi-layer threat detection pipeline.

LAYERS
------
Layer 1  Keyword    rapidfuzz multi-scorer on normalised text AND
                    sliding 1-to-3 segment windows
Layer 2  Semantic   sentence-transformers cosine similarity (optional)
Layer 3  Context    benign-signal dampening, speaker-level escalation
Layer 4  LLM judge  borderline cases only (off by default)

Entry points
------------
analyze_segment(segment, context_window)   → enriched result dict
detect_threats(segments)                   → list of enriched dicts
summarize_flags(enriched_list)             → call-level summary dict

The shape returned by analyze_segment() is backward-compatible with the
original two-field output plus the new stage-score fields:

    {
      "flag":            bool,
      "category":        str,
      "confidence":      float,      # fused 0-1
      "matched_keyword": str,
      # new fields:
      "keyword_score":   float,
      "semantic_score":  float,
      "llm_score":       float | None,
      "llm_reason":      str | None,
      "severity":        str,        # Low | Medium | High
      "stage_details":   dict,       # full per-layer breakdown
    }
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from config import cfg
from pipeline.normalize import normalize, normalize_for_matching


# ─────────────────────────────────────────────────────────────────────────────
# Keyword cache
# ─────────────────────────────────────────────────────────────────────────────

_keywords_cache: dict | None        = None
_keywords_path_cache: str | None    = None
_flat_keywords_cache: list | None   = None   # [(kw_lower, category), ...]


def load_keywords(path: str | None = None) -> dict[str, list[str]]:
    """Load keywords.json, cached in memory."""
    global _keywords_cache, _keywords_path_cache, _flat_keywords_cache
    path = path or cfg.KEYWORDS_PATH
    if _keywords_cache is not None and _keywords_path_cache == path:
        return _keywords_cache
    kw_path = Path(path)
    if not kw_path.exists():
        raise FileNotFoundError(f"Keywords file not found: {kw_path.resolve()}")
    with open(kw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _keywords_cache       = {k: v for k, v in data.items() if not k.startswith("_")}
    _keywords_path_cache  = path
    _flat_keywords_cache  = None   # force rebuild
    return _keywords_cache


def invalidate_keyword_cache() -> None:
    global _keywords_cache, _keywords_path_cache, _flat_keywords_cache
    _keywords_cache = _keywords_path_cache = _flat_keywords_cache = None


def _get_flat_keywords() -> list[tuple[str, str]]:
    global _flat_keywords_cache
    if _flat_keywords_cache is not None:
        return _flat_keywords_cache
    keywords = load_keywords()
    # Also add normalised forms so the matcher sees the same text as input
    entries: list[tuple[str, str]] = []
    for cat, kw_list in keywords.items():
        for kw in kw_list:
            norm = normalize(kw)
            entries.append((kw.lower(), cat))
            if norm != kw.lower():
                entries.append((norm, cat))
    # Deduplicate by (kw, cat)
    seen = set()
    _flat_keywords_cache = []
    for item in entries:
        if item not in seen:
            seen.add(item)
            _flat_keywords_cache.append(item)
    return _flat_keywords_cache


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: multi-scorer fuzzy keyword matching
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_match(text: str, threshold: int | None = None) -> list[dict]:
    """
    Match `text` against all keywords using three rapidfuzz scorers:
      - WRatio          (best general scorer for short texts)
      - partial_ratio   (substring match — catches keywords inside sentences)
      - token_set_ratio (order-insensitive — helps split phrases)

    Returns list of {category, keyword, score} sorted descending by score.
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        return []

    thr = threshold if threshold is not None else cfg.FUZZY_THRESHOLD
    flat = _get_flat_keywords()
    if not flat:
        return []

    kw_strings = [kw for kw, _ in flat]

    # Run all three scorers and keep the best score per (keyword, category)
    best: dict[tuple[str, str], float] = {}

    for scorer in (fuzz.WRatio, fuzz.partial_ratio, fuzz.token_set_ratio):
        for candidate, score, idx in process.extract(
            text, kw_strings, scorer=scorer, score_cutoff=thr, limit=None
        ):
            key = (flat[idx][0], flat[idx][1])
            if best.get(key, 0) < score:
                best[key] = score

    if not best:
        return []

    results = [
        {"keyword": kw, "category": cat, "score": score}
        for (kw, cat), score in best.items()
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _keyword_score_for_texts(texts: list[str]) -> list[dict]:
    """
    Run fuzzy matching across multiple candidate strings (normalised variants)
    and return the merged best-match list.
    """
    best: dict[tuple[str, str], float] = {}
    for t in texts:
        for m in fuzzy_match(t):
            key = (m["keyword"], m["category"])
            if best.get(key, 0) < m["score"]:
                best[key] = m["score"]
    results = [
        {"keyword": kw, "category": cat, "score": sc}
        for (kw, cat), sc in best.items()
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: context modifiers
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that suggest the flagged text is benign (sport, humour, work)
_BENIGN_EN = [
    "at work", "killing it", "crushed it", "nailed it",
    "soccer", "football", "basketball", "cricket",
    "game", "match", "tournament", "series",
    "movie", "film", "book", "chapter", "story",
    "joke", "kidding", "just saying", "not serious",
    "presentation", "project", "deadline", "meeting",
]
_BENIGN_HI = [
    "kaam mein", "project mein", "match mein",
    "movie mein", "film mein", "mazhak", "mazaak", "joke kar",
]
_BENIGN_ALL = _BENIGN_EN + _BENIGN_HI


def _context_modifier(
    segment: dict,
    context_window: list[dict],
    raw_score: float,
) -> tuple[float, str]:
    """
    Apply context-based adjustments to raw_score.
    Returns (modified_score, reason_string).
    """
    ctx_text = " ".join(s.get("text", "").lower() for s in context_window)
    seg_text = segment.get("text", "").lower()

    # Benign signal in context → dampen by 30 %
    if any(sig in ctx_text for sig in _BENIGN_ALL):
        return max(0.0, raw_score - 0.30), "benign_context"

    # Benign signal in the segment itself → dampen by 20 %
    if any(sig in seg_text for sig in _BENIGN_ALL):
        return max(0.0, raw_score - 0.20), "benign_self"

    # Escalation: ≥2 medium flags from same speaker in context → boost by 10 %
    spk = segment.get("speaker", "")
    same_spk_flags = [
        s for s in context_window
        if s.get("speaker") == spk and s.get("confidence", 0) >= 0.40
    ]
    if len(same_spk_flags) >= 2:
        return min(1.0, raw_score + 0.10), "speaker_escalation"

    return raw_score, "no_change"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4: LLM judge
# ─────────────────────────────────────────────────────────────────────────────

def _llm_judge(
    segment: dict,
    context_window: list[dict],
    fused_score: float,
) -> tuple[float | None, str | None]:
    """
    Call LLM judge for borderline scores.
    Returns (llm_confidence, reason) or (None, None) on failure / not applicable.
    """
    if cfg.LLM_JUDGE.lower() == "off":
        return None, None
    if not (cfg.LLM_BORDERLINE_LOW <= fused_score <= cfg.LLM_BORDERLINE_HIGH):
        return None, None

    ctx_lines = "\n".join(
        f"  {s.get('speaker','?')}: {s.get('text','')}"
        for s in context_window[-4:]
    )
    prompt = (
        "You are a threat-detection classifier. "
        "Respond ONLY with a JSON object and nothing else.\n\n"
        f"Context (previous turns):\n{ctx_lines}\n\n"
        f"Target segment ({segment.get('speaker','?')}): "
        f"{segment.get('text','')}\n\n"
        "JSON schema: "
        "{\"is_threat\": bool, \"category\": str, "
        "\"severity\": \"Low\"|\"Medium\"|\"High\", "
        "\"confidence\": float 0-1, \"reason\": str}\n"
        "Categories: threat | abuse | harm_planning | disaster | none"
    )

    try:
        if cfg.LLM_JUDGE.lower() == "ollama":
            import requests, json as _json
            resp = requests.post(
                f"{cfg.OLLAMA_HOST}/api/generate",
                json={"model": cfg.LLM_MODEL, "prompt": prompt, "stream": False},
                timeout=cfg.LLM_TIMEOUT_S,
            )
            raw = resp.json().get("response", "")
        elif cfg.LLM_JUDGE.lower() == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model   = cfg.LLM_MODEL,
                max_tokens = 256,
                messages   = [{"role": "user", "content": prompt}],
                timeout    = cfg.LLM_TIMEOUT_S,
            )
            raw = msg.content[0].text
        else:
            return None, None

        # Parse JSON from response
        import json as _json, re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None, None
        data = _json.loads(m.group())
        conf   = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))
        return conf, reason

    except Exception as exc:
        print(f"[detector] LLM judge failed: {exc}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Severity mapping
# ─────────────────────────────────────────────────────────────────────────────

def _severity(confidence: float) -> str:
    if confidence >= cfg.EMERGENCY_THRESHOLD:
        return "High"
    if confidence >= cfg.WARNING_THRESHOLD:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# SWAPPABLE ENTRY POINT: analyze_segment
# ─────────────────────────────────────────────────────────────────────────────

def analyze_segment(
    segment: dict,
    context_window: list[dict],
) -> dict:
    """
    ╔════════════════════════════════════════════════════════════════╗
    ║  SWAPPABLE INTERFACE — drop in any classifier here            ║
    ╚════════════════════════════════════════════════════════════════╝

    Input
    -----
    segment        : {speaker, text, start, end, [asr_text, asr_language, …]}
    context_window : list of preceding segment dicts

    Output (backward-compatible + new fields)
    ------------------------------------------
    {
      flag, category, confidence, matched_keyword,
      keyword_score, semantic_score,
      llm_score, llm_reason,
      severity, stage_details,
    }
    """
    text = segment.get("text", "")
    if not text.strip():
        return _empty_result()

    # ── Normalise ─────────────────────────────────────────────────────────
    candidates = normalize_for_matching(text)   # [base, deva, roman]

    # ── Layer 1: Keyword ──────────────────────────────────────────────────
    kw_matches = _keyword_score_for_texts(candidates)
    if kw_matches:
        best_kw     = kw_matches[0]
        kw_score    = best_kw["score"] / 100.0
        kw_category = best_kw["category"]
        kw_word     = best_kw["keyword"]
    else:
        kw_score = 0.0
        kw_category = ""
        kw_word = ""

    # ── Layer 2: Semantic ─────────────────────────────────────────────────
    sem_result  = _try_semantic(text)
    sem_score   = sem_result.score
    sem_cat     = sem_result.category
    sem_phrase  = sem_result.nearest_phrase

    # ── Choose dominant category ──────────────────────────────────────────
    # Keyword wins on ties; semantic overrides when it is much stronger
    if kw_score >= sem_score or not sem_cat:
        category = kw_category or sem_cat
    else:
        category = sem_cat

    if not category:
        return _empty_result()

    # ── Fuse scores (weighted average) ───────────────────────────────────
    total_weight = cfg.WEIGHT_KEYWORD + cfg.WEIGHT_SEMANTIC
    fused = (
        cfg.WEIGHT_KEYWORD  * kw_score
        + cfg.WEIGHT_SEMANTIC * sem_score
    ) / total_weight if total_weight else kw_score

    # ── Layer 3: Context modifier ─────────────────────────────────────────
    fused, ctx_reason = _context_modifier(segment, context_window, fused)

    # ── Layer 4: LLM judge ────────────────────────────────────────────────
    llm_score, llm_reason = _llm_judge(segment, context_window, fused)
    if llm_score is not None:
        # LLM verdict overrides fused score
        fused     = llm_score
        category_from_llm = llm_reason  # reason is a string, not the category

    # ── Final flag decision ───────────────────────────────────────────────
    flagged    = fused >= cfg.FLAG_THRESHOLD
    severity   = _severity(fused)
    confidence = round(fused, 4)

    stage_details = {
        "keyword_score":   round(kw_score, 4),
        "keyword_match":   kw_word,
        "keyword_category": kw_category,
        "semantic_score":  round(sem_score, 4),
        "semantic_category": sem_cat,
        "nearest_phrase":  sem_phrase,
        "semantic_available": sem_result.available,
        "context_reason":  ctx_reason,
        "fused_score":     confidence,
        "llm_called":      llm_score is not None,
        "llm_score":       round(llm_score, 4) if llm_score is not None else None,
        "llm_reason":      llm_reason,
        "flag_threshold":  cfg.FLAG_THRESHOLD,
    }

    return {
        "flag":             flagged,
        "category":         category if flagged else "",
        "confidence":       confidence,
        "matched_keyword":  kw_word if flagged else "",
        "keyword_score":    round(kw_score, 4),
        "semantic_score":   round(sem_score, 4),
        "llm_score":        round(llm_score, 4) if llm_score is not None else None,
        "llm_reason":       llm_reason,
        "severity":         severity,
        "stage_details":    stage_details,
    }


def _empty_result() -> dict:
    return {
        "flag": False, "category": "", "confidence": 0.0,
        "matched_keyword": "", "keyword_score": 0.0,
        "semantic_score": 0.0, "llm_score": None,
        "llm_reason": None, "severity": "Low",
        "stage_details": {},
    }


def _try_semantic(text: str):
    """Wrapper that catches import errors gracefully."""
    try:
        from pipeline.semantic import semantic_score
        return semantic_score(text)
    except Exception:
        from pipeline.semantic import SemanticResult
        return SemanticResult(0.0, "", "", False)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window segment-boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def _window_text(segments: list[dict], idx: int, width: int) -> str:
    """Concatenate text of `width` segments ending at idx (inclusive)."""
    start = max(0, idx - width + 1)
    return " ".join(s.get("text", "") for s in segments[start: idx + 1])


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def detect_threats(segments: list[dict]) -> list[dict]:
    """
    Run all detection layers over a full transcript.

    Also checks 2- and 3-segment sliding windows so threats split across
    segment boundaries are caught.

    Returns an enriched copy of each segment dict.
    """
    results: list[dict] = []

    for i, segment in enumerate(segments):
        window_start   = max(0, i - cfg.CONTEXT_WINDOW_SIZE)
        context_window = segments[window_start:i]

        # ── Single-segment analysis ───────────────────────────────────────
        analysis = analyze_segment(segment, context_window)

        # ── Sliding window pass (2- and 3-segment) ────────────────────────
        # Only run if single-segment produced no strong flag
        if not analysis["flag"] or analysis["confidence"] < 0.70:
            for width in (2, 3):
                if i < width - 1:
                    continue
                win_text = _window_text(segments, i, width)
                win_seg  = {**segment, "text": win_text}
                win_analysis = analyze_segment(win_seg, context_window)
                # Promote if the window yields a stronger hit
                if win_analysis["confidence"] > analysis["confidence"]:
                    # Attribute the flag to the current (last) segment
                    analysis = win_analysis

        enriched = {
            **segment,
            **analysis,
            "context_window": [
                {"speaker": s.get("speaker", ""), "text": s.get("text", "")}
                for s in context_window
            ],
        }
        results.append(enriched)

    return results


def summarize_flags(results: list[dict]) -> dict:
    """Produce call-level summary statistics."""
    flagged = [r for r in results if r.get("flag")]
    by_category: dict[str, int] = {}
    for r in flagged:
        cat = r.get("category", "")
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "total_segments":     len(results),
        "total_flags":        len(flagged),
        "by_category":        by_category,
        "highest_confidence": max((r["confidence"] for r in flagged), default=0.0),
        "flagged_segments":   flagged,
    }


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE = [
        {"speaker": "A", "text": "Hey, how's the project going?", "start": 0.0, "end": 2.5},
        {"speaker": "B", "text": "We're totally killing it at work!", "start": 2.6, "end": 5.1},
        {"speaker": "A", "text": "We need to get rid of him. Nobody will find out.", "start": 5.2, "end": 9.0},
        {"speaker": "B", "text": "I'll kill you if you breathe a word.", "start": 9.1, "end": 12.0},
        {"speaker": "A", "text": "Let's blow up the presentation tomorrow.", "start": 12.1, "end": 15.0},
        {"speaker": "B", "text": "maar dunga tujhe agar tune kisi ko bataya", "start": 15.1, "end": 18.5},
        {"speaker": "A", "text": "مار دوں گا تمہیں", "start": 18.6, "end": 20.0},
    ]
    results  = detect_threats(SAMPLE)
    summary  = summarize_flags(results)
    for r in results:
        marker = " ⚠ FLAGGED" if r["flag"] else ""
        print(f"[{r['start']:5.1f}s] {r['speaker']}: {r['text']}{marker}")
        if r["flag"]:
            d = r.get("stage_details", {})
            print(f"         cat={r['category']} conf={r['confidence']:.2f} sev={r['severity']}")
            print(f"         kw={d.get('keyword_score',0):.2f} sem={d.get('semantic_score',0):.2f}")
    print(f"\nTotal flags: {summary['total_flags']}/{summary['total_segments']}")

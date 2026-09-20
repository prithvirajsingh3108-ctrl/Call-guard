"""
pipeline/scoring.py
────────────────────
risk_score(flags) — compute a 0–100 integer risk score for a call.

Formula
-------
  score = round(0.7 × top_confidence + 0.3 × second_confidence) × 100

  - top_confidence    : highest flag confidence in the call  (0.0–1.0)
  - second_confidence : second-highest flag confidence        (0.0–1.0, 0 if only one flag)
  - Returns 0 when there are no flags.
  - Result is clamped to the 0–100 range.
"""

from __future__ import annotations


def risk_score(flags: list[dict]) -> int:
    """
    Compute a 0–100 integer risk score from a list of flag dicts.

    Each flag dict must contain a 'confidence' key (float 0.0–1.0).
    Accepts both live flag dicts (from detector.detect_threats) and
    flat dicts built from DB Flag ORM objects.

    Returns
    -------
    int
        0   — no flags
        1-29 — Low risk
        30-69 — Medium risk
        70-100 — High risk
    """
    if not flags:
        return 0

    # Gather confidence values, filter out None / invalid
    confidences = []
    for f in flags:
        if isinstance(f, dict):
            c = f.get("confidence")
        else:
            # ORM Flag object
            c = getattr(f, "confidence", None)
        if c is not None:
            try:
                confidences.append(float(c))
            except (TypeError, ValueError):
                pass

    if not confidences:
        return 0

    confidences.sort(reverse=True)
    top    = confidences[0]
    second = confidences[1] if len(confidences) > 1 else 0.0

    raw = 0.7 * top + 0.3 * second
    return max(0, min(100, round(raw * 100)))

"""
pipeline/normalize.py
─────────────────────
Text normalisation pipeline for multi-lingual threat detection.

normalize(text) is the single public entry point.  It applies steps in order:

  1. Unicode NFKC decomposition
  2. Lowercase
  3. Urdu/Arabic glyph normalisation  (ي→ی, ك→ک, ه→ہ, etc.)
  4. Devanagari nukta normalisation   (क़→क, ख़→ख, …)
  5. Strip punctuation / non-word chars (keep spaces and script letters)
  6. Collapse repeated characters      ("maaaar" → "maar", "killll" → "kill")
  7. Roman Hindi/Urdu spelling-variant map  ("maar" → "mar", "dunga" → "dunga")
  8. Roman → Devanagari transliteration token overlay
     Adds the Devanagari form of each Roman token so the keyword matcher
     can hit both forms without needing duplicate entries in keywords.json.

normalize_for_matching(text) returns a list of candidate strings:
  [original normalised, devanagari overlay, back-transliterated form]
  so the detector can match all representations in one pass.
"""

from __future__ import annotations

import re
import unicodedata

# ─────────────────────────────────────────────────────────────────────────────
# Urdu / Arabic glyph variants → canonical Urdu form
# ─────────────────────────────────────────────────────────────────────────────
_URDU_MAP: dict[str, str] = {
    "\u0643": "\u06A9",  # ك → ک  Arabic kaf  → Urdu kaf
    "\u064A": "\u06CC",  # ي → ی  Arabic yeh  → Urdu yeh
    "\u0647": "\u06BE",  # ه → ھ  Arabic heh  → Urdu do chashmi heh
    "\u0629": "\u06C1",  # ة → ہ  Arabic teh marbuta → Urdu heh goal
    "\u0649": "\u06CC",  # ى → ی  Arabic alef maqsura → Urdu yeh
    "\u0671": "\u0627",  # ٱ → ا  Arabic wasla alef → plain alef
    "\u0622": "\u0627",  # آ → ا  Arabic alef madda → plain alef (for loose matching)
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
}

def _normalise_urdu(text: str) -> str:
    return "".join(_URDU_MAP.get(ch, ch) for ch in text)


# ─────────────────────────────────────────────────────────────────────────────
# Devanagari nukta normalisation
# Nukta (U+093C) combines with a consonant to form the dotted variant.
# For fuzzy matching we strip nukta so क़/क both match the same keyword.
# ─────────────────────────────────────────────────────────────────────────────
_NUKTA = "\u093C"  # ़

# Pre-composed dotted forms → base consonant
_NUKTA_MAP: dict[str, str] = {
    "\u0958": "\u0915",  # क़ → क
    "\u0959": "\u0916",  # ख़ → ख
    "\u095A": "\u0917",  # ग़ → ग
    "\u095B": "\u091C",  # ज़ → ज
    "\u095C": "\u0921",  # ड़ → ड
    "\u095D": "\u0922",  # ढ़ → ढ
    "\u095E": "\u092B",  # फ़ → फ
    "\u095F": "\u092F",  # य़ → य
}

def _normalise_devanagari(text: str) -> str:
    for composed, base in _NUKTA_MAP.items():
        text = text.replace(composed, base)
    return text.replace(_NUKTA, "")  # strip any remaining combining nukta


# ─────────────────────────────────────────────────────────────────────────────
# Collapse repeated characters  ("maaaar" → "maar", "killll" → "kil")
# Keep at most 2 consecutive identical characters so "maar" stays "maar"
# but "maaaar" collapses to "maar".
# ─────────────────────────────────────────────────────────────────────────────
_RE_REPEAT = re.compile(r"(.)\1{2,}")

def _collapse_repeats(text: str) -> str:
    return _RE_REPEAT.sub(r"\1\1", text)


# ─────────────────────────────────────────────────────────────────────────────
# Roman Hindi/Urdu spelling-variant normalisation
# Maps common alternate spellings to a canonical Roman form so both
# "maar dunga" and "mar dunga" match the same keyword entry.
# ─────────────────────────────────────────────────────────────────────────────
_ROMAN_VARIANTS: list[tuple[str, str]] = [
    # verb endings
    (r"\bdunga\b",   "dunga"),
    (r"\bdoonga\b",  "dunga"),
    (r"\bdonga\b",   "dunga"),
    (r"\bdenge\b",   "denge"),
    (r"\bdengey\b",  "denge"),
    # kill / hurt variants
    (r"\bmaar\b",    "mar"),
    (r"\bmarega\b",  "marega"),
    (r"\bmardunga\b","mar dunga"),
    (r"\bmardoonga\b","mar dunga"),
    # khatam variants
    (r"\bkhatam\b",  "khatam"),
    (r"\bkhatm\b",   "khatam"),
    # jaan variants
    (r"\bjaan\b",    "jaan"),
    (r"\bjan\b",     "jaan"),
    # goli variants
    (r"\bgoli\b",    "goli"),
    (r"\bguly\b",    "goli"),
    # chaku / chaqoo
    (r"\bchaqoo\b",  "chaku"),
    (r"\bchaqu\b",   "chaku"),
    (r"\bchakku\b",  "chaku"),
    # barbad
    (r"\bbarbad\b",  "barbad"),
    (r"\bbarbaad\b", "barbad"),
    # tujhe / tumhe
    (r"\btujhe\b",   "tujhe"),
    (r"\btujhey\b",  "tujhe"),
    (r"\btumhe\b",   "tumhe"),
    (r"\btumhey\b",  "tumhe"),
    # usse / use
    (r"\busse\b",    "use"),
    (r"\busey\b",    "use"),
    # saboot / sabit
    (r"\bsaboot\b",  "saboot"),
    (r"\bsabuut\b",  "saboot"),
    # common Urdu-Roman
    (r"\bqatl\b",    "qatl"),
    (r"\bkatal\b",   "qatl"),
    (r"\bqatal\b",   "qatl"),
    (r"\bzeher\b",   "zeher"),
    (r"\bzehr\b",    "zeher"),
    (r"\bzehar\b",   "zeher"),
    # dhamaka
    (r"\bdhamaka\b", "dhamaka"),
    (r"\bdhamakka\b","dhamaka"),
    (r"\bdhamake\b", "dhamaka"),
    # imaarat
    (r"\bimaarat\b", "imaarat"),
    (r"\bimarat\b",  "imaarat"),
]

# Compile once
_ROMAN_VARIANT_RES: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _ROMAN_VARIANTS
]

def _normalise_roman_variants(text: str) -> str:
    for pattern, replacement in _ROMAN_VARIANT_RES:
        text = pattern.sub(replacement, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Minimal Roman → Devanagari token map
# Maps common Roman Hindi/Urdu threat tokens to their Devanagari equivalents.
# This is intentionally narrow — we only map tokens that appear in keywords.json
# so the keyword matcher can hit both scripts.
# ─────────────────────────────────────────────────────────────────────────────
_ROMAN_TO_DEVA: dict[str, str] = {
    "mar":        "मार",
    "maar":       "मार",
    "dunga":      "दूंगा",
    "denge":      "देंगे",
    "jaan":       "जान",
    "goli":       "गोली",
    "chaku":      "चाकू",
    "khatam":     "खत्म",
    "barbad":     "बर्बाद",
    "tujhe":      "तुझे",
    "tumhe":      "तुम्हें",
    "marna":      "मारना",
    "marunga":    "मारूंगा",
    "marega":     "मारेगा",
    "khatamkar":  "खत्मकर",
    "saboot":     "सबूत",
    "gaayab":     "गायब",
    "tabah":      "तबाह",
    "zeher":      "ज़हर",
    "bomb":       "बम",
    "dhamaka":    "धमाका",
    "imaarat":    "इमारत",
    "qatl":       "क़त्ल",
    "nipta":      "निपटा",
    "hatao":      "हटाओ",
    "chhod":      "छोड़",
    "dekh":       "देख",
    "raste":      "रास्ते",
    "kaam":       "काम",
    "tamam":      "तमाम",
}

# Devanagari → Roman (reverse map, used for back-transliteration check)
_DEVA_TO_ROMAN: dict[str, str] = {v: k for k, v in _ROMAN_TO_DEVA.items()}


def _roman_to_devanagari_overlay(text: str) -> str:
    """
    Replace recognised Roman tokens with their Devanagari form.
    Returns a new string that can be compared against Devanagari keywords.
    """
    tokens = text.split()
    out = []
    for t in tokens:
        out.append(_ROMAN_TO_DEVA.get(t, t))
    return " ".join(out)


def _devanagari_to_roman_overlay(text: str) -> str:
    """Replace Devanagari tokens with Roman equivalents."""
    tokens = text.split()
    out = []
    for t in tokens:
        out.append(_DEVA_TO_ROMAN.get(t, t))
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Strip punctuation — keep Unicode letters, digits, spaces
# ─────────────────────────────────────────────────────────────────────────────
_RE_STRIP = re.compile(r"[^\w\s]", re.UNICODE)
_RE_SPACE = re.compile(r"\s+")

def _strip_punct(text: str) -> str:
    text = _RE_STRIP.sub(" ", text)
    return _RE_SPACE.sub(" ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Full normalisation pipeline for a single text string.

    Returns a single normalised string suitable for keyword matching.
    Steps: NFKC → lowercase → Urdu glyphs → Devanagari nukta →
           strip punct → collapse repeats → Roman variant map.
    """
    if not text:
        return ""
    # 1. Unicode NFKC
    text = unicodedata.normalize("NFKC", text)
    # 2. Lowercase
    text = text.lower()
    # 3. Urdu/Arabic glyph normalisation
    text = _normalise_urdu(text)
    # 4. Devanagari nukta
    text = _normalise_devanagari(text)
    # 5. Strip punctuation
    text = _strip_punct(text)
    # 6. Collapse repeated chars
    text = _collapse_repeats(text)
    # 7. Roman variant map
    text = _normalise_roman_variants(text)
    return text


def normalize_for_matching(text: str) -> list[str]:
    """
    Return a list of candidate strings for matching:
      [0] normalised original text
      [1] Roman tokens replaced with Devanagari
      [2] Devanagari tokens replaced with Roman

    The detector uses all three so "mar dunga", "मार दूंगा" and
    "مار دوں گا" all converge on the same keyword hits.
    """
    base = normalize(text)
    deva = normalize(_roman_to_devanagari_overlay(base))
    roman = normalize(_devanagari_to_roman_overlay(base))
    # Deduplicate while preserving order
    seen: list[str] = []
    for s in [base, deva, roman]:
        if s and s not in seen:
            seen.append(s)
    return seen


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "I'll KILL you!!",
        "maaaar dunga tujhe",
        "maar dunga",
        "جان سے مار دوں گا",
        "मैं तुम्हें मार दूंगा",
        "We need to get rid of him, nobody will find out",
        "killing it at work this week",
        "Let's blow up the presentation tomorrow",
        "qatl karunga usse",
        "Zeher mila do paani mein",
    ]
    for t in tests:
        candidates = normalize_for_matching(t)
        print(f"\nInput:      {t!r}")
        for i, c in enumerate(candidates):
            print(f"  Form {i}:  {c!r}")

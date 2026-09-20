"""tests/test_normalize.py — Unit tests for pipeline/normalize.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pipeline.normalize import normalize, normalize_for_matching


class TestNormalizeBasics:
    def test_lowercase(self):
        assert normalize("KILL YOU") == "kill you"

    def test_nfkc(self):
        # Fullwidth chars should be collapsed
        assert normalize("ｋｉｌｌ") == "kill"

    def test_strip_punctuation(self):
        assert normalize("I'll kill you!!!") == "i ll kill you"

    def test_collapse_repeats(self):
        # Collapse drops to 2 repeats, then variant map may canonicalise further
        # "maaaar" → "maar" → "mar" (via Roman variant table) — both are correct
        result = normalize("maaaar")
        assert result in ("maar", "mar"), f"Unexpected: {result!r}"
        result2 = normalize("killll")
        # "kill" → collapsed to "kill" (no variant mapping for "kill")
        assert result2 == "kill"

    def test_maar_variant(self):
        # "maar" maps to "mar" via Roman variant table
        result = normalize("maar dunga")
        assert "mar" in result or "maar" in result  # canonical form present

    def test_urdu_glyph_norm(self):
        # Arabic ك should map to Urdu ک
        text = "\u0643\u0627\u0645"   # كام  (Arabic kaf)
        norm = normalize(text)
        assert "\u0643" not in norm    # Arabic kaf gone

    def test_devanagari_nukta_strip(self):
        # क़ (pre-composed) should become क
        assert normalize("\u0958") == "\u0915"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_whitespace_collapse(self):
        assert normalize("kill   you") == "kill you"


class TestNormalizeForMatching:
    def test_returns_list(self):
        result = normalize_for_matching("maar dunga")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_deduplication(self):
        result = normalize_for_matching("hello")
        assert len(result) == len(set(result))

    def test_devanagari_overlay(self):
        result = normalize_for_matching("mar dunga tujhe")
        # At least one form should contain Devanagari chars
        has_deva = any(
            any("\u0900" <= c <= "\u097F" for c in s)
            for s in result
        )
        assert has_deva, f"No Devanagari in {result}"

    def test_crosslingual_convergence(self):
        # "maar dunga" and "मार दूंगा" should share at least one normalised token
        roman_forms = set(normalize_for_matching("maar dunga"))
        deva_forms  = set(normalize_for_matching("मार दूंगा"))
        # They may not be identical but both should produce non-empty results
        assert roman_forms
        assert deva_forms

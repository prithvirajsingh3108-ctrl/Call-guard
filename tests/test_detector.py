"""tests/test_detector.py — Unit tests for pipeline/detector.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pipeline.detector import (
    analyze_segment, detect_threats, summarize_flags,
    fuzzy_match, load_keywords, invalidate_keyword_cache,
)


def seg(text, speaker="A", start=0.0, end=1.0):
    return {"speaker": speaker, "text": text, "start": start, "end": end}


class TestFuzzyMatch:
    def test_clear_english_threat(self):
        matches = fuzzy_match("i will kill you")
        assert matches, "Expected at least one match"
        assert matches[0]["category"] == "threat"

    def test_no_match_for_benign(self):
        # "the weather is nice today" should not match any threat keyword above threshold
        # when using WRatio (token_set_ratio can produce false hits on very short keywords)
        from rapidfuzz import fuzz
        text = "the weather is nice today"
        matches = fuzzy_match(text, threshold=85)
        # Only assert no matches above a stricter threshold
        assert not matches or matches[0]["score"] < 90, \
            f"Unexpected high-confidence match on benign text: {matches[:3]}"

    def test_roman_hindi_threat(self):
        matches = fuzzy_match("maar dunga tujhe")
        assert matches, "Expected match for Roman Hindi threat"

    def test_threshold_respected(self):
        # Score cutoff should filter weak matches
        matches = fuzzy_match("kill", threshold=90)
        for m in matches:
            assert m["score"] >= 90

    def test_returns_sorted_descending(self):
        matches = fuzzy_match("i will kill you")
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i]["score"] >= matches[i + 1]["score"]


class TestAnalyzeSegment:
    def test_english_threat_flagged(self):
        result = analyze_segment(seg("I will kill you"), [])
        assert result["flag"] is True
        assert result["category"] == "threat"
        assert 0.0 < result["confidence"] <= 1.0

    def test_roman_hindi_threat_flagged(self):
        result = analyze_segment(seg("maar dunga tujhe agar bataya"), [])
        assert result["flag"] is True

    def test_devanagari_threat_flagged(self):
        result = analyze_segment(seg("मैं तुम्हें मार दूंगा"), [])
        assert result["flag"] is True

    def test_urdu_threat_flagged(self):
        result = analyze_segment(seg("تمہیں مار دوں گا"), [])
        assert result["flag"] is True

    def test_benign_work_context_dampened(self):
        ctx = [
            seg("killing it at work"),
            seg("the presentation was amazing"),
        ]
        result = analyze_segment(seg("I'll kill this presentation"), ctx)
        # Should either not flag or flag with reduced confidence
        if result["flag"]:
            assert result["confidence"] < 0.70, "Benign context should dampen score"

    def test_empty_text_not_flagged(self):
        result = analyze_segment(seg(""), [])
        assert result["flag"] is False

    def test_result_shape(self):
        result = analyze_segment(seg("some text"), [])
        required = ["flag", "category", "confidence", "matched_keyword",
                    "keyword_score", "semantic_score", "severity", "stage_details"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_confidence_in_range(self):
        result = analyze_segment(seg("I will kill you"), [])
        assert 0.0 <= result["confidence"] <= 1.0

    def test_severity_mapping(self):
        result = analyze_segment(seg("I will kill you"), [])
        assert result["severity"] in ("Low", "Medium", "High")

    def test_harm_planning(self):
        result = analyze_segment(
            seg("we need to get rid of him nobody will find out"), []
        )
        assert result["flag"] is True
        assert result["category"] == "harm_planning"

    def test_disaster(self):
        result = analyze_segment(seg("let's blow up the building"), [])
        assert result["flag"] is True
        assert result["category"] == "disaster"

    def test_abuse(self):
        result = analyze_segment(seg("you are worthless and pathetic"), [])
        assert result["flag"] is True
        assert result["category"] == "abuse"


class TestDetectThreats:
    def test_returns_same_length(self):
        segments = [seg("hello"), seg("I will kill you"), seg("ok bye")]
        results  = detect_threats(segments)
        assert len(results) == len(segments)

    def test_enriched_fields_present(self):
        results = detect_threats([seg("I will kill you")])
        r = results[0]
        assert "flag" in r
        assert "context_window" in r

    def test_context_window_built(self):
        segments = [
            seg("hello there"),
            seg("how are you"),
            seg("I will kill you"),
        ]
        results = detect_threats(segments)
        flagged = [r for r in results if r["flag"]]
        assert flagged
        # context_window of the flagged segment should contain prior segs
        assert len(flagged[0]["context_window"]) >= 2

    def test_no_false_flags_on_all_benign(self):
        benign = [
            seg("killing it at work"),
            seg("let's crush it in the game"),
            seg("I could die for coffee right now"),
        ]
        results = detect_threats(benign)
        # All should have low or no flags after context dampening
        flagged = [r for r in results if r["flag"]]
        assert len(flagged) == 0, f"Unexpected flags: {[r['text'] for r in flagged]}"


class TestSummarizeFlags:
    def test_empty_list(self):
        s = summarize_flags([])
        assert s["total_segments"] == 0
        assert s["total_flags"] == 0

    def test_counts_flags(self):
        results = detect_threats([
            seg("I will kill you"),
            seg("nice weather today"),
            seg("you are worthless"),
        ])
        s = summarize_flags(results)
        assert s["total_segments"] == 3
        assert s["total_flags"] >= 1

    def test_by_category_populated(self):
        results = detect_threats([seg("I will kill you")])
        s = summarize_flags(results)
        assert "threat" in s["by_category"]

    def test_highest_confidence(self):
        results = detect_threats([seg("I will kill you")])
        s = summarize_flags(results)
        assert 0.0 < s["highest_confidence"] <= 1.0

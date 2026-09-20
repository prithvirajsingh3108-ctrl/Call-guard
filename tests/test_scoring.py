"""tests/test_scoring.py — Unit tests for pipeline/scoring.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pipeline.scoring import risk_score


class TestRiskScore:
    def test_no_flags(self):
        assert risk_score([]) == 0

    def test_single_full_confidence(self):
        flags = [{"confidence": 1.0}]
        assert risk_score(flags) == 70   # 0.7*1.0 + 0.3*0.0 = 0.7 → 70

    def test_two_full_confidence(self):
        flags = [{"confidence": 1.0}, {"confidence": 1.0}]
        assert risk_score(flags) == 100  # 0.7*1.0 + 0.3*1.0 = 1.0 → 100

    def test_formula(self):
        flags = [{"confidence": 0.8}, {"confidence": 0.5}, {"confidence": 0.3}]
        expected = round((0.7 * 0.8 + 0.3 * 0.5) * 100)
        assert risk_score(flags) == expected

    def test_single_half_confidence(self):
        flags = [{"confidence": 0.5}]
        # 0.7*0.5 + 0.3*0.0 = 0.35 → 35
        assert risk_score(flags) == 35

    def test_clamp_to_100(self):
        flags = [{"confidence": 2.0}, {"confidence": 1.5}]  # out-of-range input
        assert risk_score(flags) <= 100

    def test_clamp_to_0(self):
        assert risk_score([{"confidence": 0.0}]) == 0

    def test_none_confidence_skipped(self):
        flags = [{"confidence": None}, {"confidence": 0.9}]
        # None should be skipped gracefully
        result = risk_score(flags)
        assert 0 <= result <= 100

    def test_orm_object_like(self):
        """Accepts objects with a .confidence attribute (ORM Flag)."""
        class FakeFlag:
            confidence = 0.75
        result = risk_score([FakeFlag()])
        assert result == round(0.7 * 0.75 * 100)

    def test_low_medium_high_thresholds(self):
        assert risk_score([{"confidence": 0.20}]) < 30            # Low
        assert 30 <= risk_score([{"confidence": 0.50}, {"confidence": 0.40}]) < 70  # Medium
        flags_high = [{"confidence": 1.0}, {"confidence": 0.8}]
        assert risk_score(flags_high) >= 70                        # High

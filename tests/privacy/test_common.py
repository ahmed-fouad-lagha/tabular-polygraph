from tabular_polygraph.privacy.common import (
    risk_level_linkability,
    risk_level_membership,
    risk_level_singling_out,
)


class TestRiskLevel:
    def test_membership_boundaries(self):
        assert risk_level_membership(0.0) == "very_low"
        assert risk_level_membership(0.51) == "very_low"
        assert risk_level_membership(0.52) == "low"
        assert risk_level_membership(0.59) == "low"
        assert risk_level_membership(0.60) == "medium"
        assert risk_level_membership(0.69) == "medium"
        assert risk_level_membership(0.70) == "high"
        assert risk_level_membership(0.79) == "high"
        assert risk_level_membership(0.80) == "very_high"
        assert risk_level_membership(1.0) == "very_high"

    def test_singling_out_boundaries(self):
        assert risk_level_singling_out(0.0) == "very_low"
        assert risk_level_singling_out(0.0005) == "very_low"
        assert risk_level_singling_out(0.001) == "low"
        assert risk_level_singling_out(0.005) == "low"
        assert risk_level_singling_out(0.01) == "medium"
        assert risk_level_singling_out(0.04) == "medium"
        assert risk_level_singling_out(0.05) == "high"
        assert risk_level_singling_out(0.14) == "high"
        assert risk_level_singling_out(0.15) == "very_high"
        assert risk_level_singling_out(1.0) == "very_high"

    def test_linkability_boundaries(self):
        assert risk_level_linkability(0.0) == "very_low"
        assert risk_level_linkability(0.51) == "very_low"
        assert risk_level_linkability(0.52) == "low"
        assert risk_level_linkability(0.59) == "low"
        assert risk_level_linkability(0.60) == "medium"
        assert risk_level_linkability(0.69) == "medium"
        assert risk_level_linkability(0.70) == "high"
        assert risk_level_linkability(0.84) == "high"
        assert risk_level_linkability(0.85) == "very_high"
        assert risk_level_linkability(1.0) == "very_high"

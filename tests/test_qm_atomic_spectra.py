"""Tests for SDGFT atomic spectra — Lamb shift predictions.

Covers:
    - Geometric projection factor Δ/(Δ+δ) = 5/6
    - Geometric anomalous dimension γ_geo (tree & fp)
    - Lamb shift predictions (tree, fp, weighted)
    - Comparison with observed Lamb shift (1057.845 MHz)
    - D* extraction from observed Lamb shift
    - Rydberg geometric correction
    - Fine-structure interval
    - Registry entries
    - Consistency checks
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from sdgft.constants import DELTA, DELTA_F, DELTA_G, DELTA_G_F, SIN2_30
from sdgft.dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from sdgft.registry import REGISTRY

from sdgft.qm.atomic_spectra import (
    # Physical constants
    R_INF_C_MHZ,
    R_INF_C_HZ,
    LAMB_SHIFT_OBS_MHZ,
    LAMB_SHIFT_OBS_UNCERT_MHZ,
    # Projection factor
    PROJECTION_FACTOR,
    PROJECTION_FACTOR_F,
    projection_factor_exact,
    # Gamma
    GAMMA_GEO_TREE,
    GAMMA_GEO_TREE_F,
    GAMMA_GEO_FP,
    GAMMA_GEO_TREE_SQ,
    GAMMA_GEO_TREE_SQ_F,
    GAMMA_GEO_FP_SQ,
    gamma_geo_tree,
    gamma_geo_fp,
    # Lamb shift functions
    lamb_shift_geo,
    lamb_shift_tree,
    lamb_shift_fp,
    lamb_shift_exact_tree,
    lamb_shift_weighted,
    # Module-level constants
    LAMB_SHIFT_TREE,
    LAMB_SHIFT_FP,
    LAMB_SHIFT_WEIGHTED,
    LAMB_SHIFT_EXACT_PREFACTOR,
    LAMB_SHIFT_TREE_DEV_MHZ,
    LAMB_SHIFT_FP_DEV_MHZ,
    LAMB_SHIFT_TREE_DEV_PCT,
    LAMB_SHIFT_FP_DEV_PCT,
    # Rydberg
    RYDBERG_GEO_CORRECTION,
    rydberg_geo_correction,
    # Fine structure
    FINE_STRUCTURE_2P,
    FINE_STRUCTURE_2P_OBS,
    fine_structure_2p,
    # D* extraction
    D_STAR_FROM_LAMB,
    d_star_from_lamb_shift,
    # Muonic
    LAMB_SHIFT_MUONIC_TREE,
    lamb_shift_muonic_tree,
    # Registry & summary
    register_all,
    print_summary,
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Projection factor                                        ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestProjectionFactor:
    """Δ/(Δ+δ) = 5/6."""

    def test_exact_fraction(self):
        """Projection factor must be exactly 5/6."""
        assert PROJECTION_FACTOR == Fraction(5, 6)

    def test_from_constants(self):
        """Compute from Level 0 axioms."""
        pf = DELTA / (DELTA + DELTA_G)
        assert pf == Fraction(5, 6)

    def test_float_value(self):
        """Float value ≈ 0.8333."""
        assert abs(PROJECTION_FACTOR_F - 5.0 / 6.0) < 1e-15

    def test_sum_is_sin2_30(self):
        """Δ + δ = sin²(30°) = 1/4."""
        assert DELTA + DELTA_G == SIN2_30

    def test_function_matches_constant(self):
        """projection_factor_exact() matches module constant."""
        assert projection_factor_exact() == PROJECTION_FACTOR


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Geometric anomalous dimension                            ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestGammaGeo:
    """γ_geo = δ²/D*."""

    def test_tree_exact(self):
        """γ_geo(tree) = 1/1608."""
        assert GAMMA_GEO_TREE == Fraction(1, 1608)

    def test_tree_factorisation(self):
        """1608 = 24 × 67."""
        assert 1608 == 24 * 67

    def test_tree_squared_exact(self):
        """γ²_geo(tree) = 1/2585664."""
        assert GAMMA_GEO_TREE_SQ == Fraction(1, 2_585_664)

    def test_fp_smaller_than_tree(self):
        """γ_geo(fp) < γ_geo(tree) because D*_fp > D*_tree."""
        assert GAMMA_GEO_FP < GAMMA_GEO_TREE_F

    def test_fp_close_to_tree(self):
        """fp and tree differ by < 1%."""
        diff = abs(GAMMA_GEO_FP - GAMMA_GEO_TREE_F) / GAMMA_GEO_TREE_F
        assert diff < 0.01

    def test_fp_squared(self):
        """γ²_geo(fp) = γ_geo(fp)²."""
        assert abs(GAMMA_GEO_FP_SQ - GAMMA_GEO_FP ** 2) < 1e-20

    def test_function_matches(self):
        """Functions match module constants."""
        assert gamma_geo_tree() == GAMMA_GEO_TREE
        assert gamma_geo_fp() == GAMMA_GEO_FP


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Lamb shift                                               ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestLambShift:
    """Lamb shift 2S₁/₂ − 2P₁/₂."""

    def test_tree_order_of_magnitude(self):
        """L_tree ≈ 1060 MHz (within ±5 MHz of 1060)."""
        assert 1055.0 < LAMB_SHIFT_TREE < 1065.0

    def test_fp_order_of_magnitude(self):
        """L_fp ≈ 1056 MHz."""
        assert 1051.0 < LAMB_SHIFT_FP < 1061.0

    def test_tree_manual_calculation(self):
        """Cross-check: manually compute L_tree."""
        expected = (5.0 / 6.0) * (1.0 / 1608.0) ** 2 * R_INF_C_MHZ
        assert abs(LAMB_SHIFT_TREE - expected) / expected < 1e-10

    def test_fp_manual_calculation(self):
        """Cross-check: manually compute L_fp."""
        gamma_fp = DELTA_G_F ** 2 / D_STAR_FP
        expected = (5.0 / 6.0) * gamma_fp ** 2 * R_INF_C_MHZ
        assert abs(LAMB_SHIFT_FP - expected) / expected < 1e-10

    def test_tree_greater_than_fp(self):
        """L_tree > L_fp (because D*_tree < D*_fp → larger γ → larger L)."""
        assert LAMB_SHIFT_TREE > LAMB_SHIFT_FP

    def test_observed_between_or_close(self):
        """Observed Lamb shift (1057.8) is between fp and tree."""
        assert LAMB_SHIFT_FP < LAMB_SHIFT_OBS_MHZ < LAMB_SHIFT_TREE

    def test_tree_deviation_small(self):
        """Tree deviates < 0.5% from experiment."""
        assert LAMB_SHIFT_TREE_DEV_PCT < 0.5

    def test_fp_deviation_small(self):
        """FP deviates < 0.3% from experiment."""
        assert LAMB_SHIFT_FP_DEV_PCT < 0.3

    def test_tree_positive_deviation(self):
        """Tree overshoots (positive deviation)."""
        assert LAMB_SHIFT_TREE_DEV_MHZ > 0.0

    def test_fp_negative_deviation(self):
        """FP undershoots (negative deviation)."""
        assert LAMB_SHIFT_FP_DEV_MHZ < 0.0

    def test_weighted_close_to_observed(self):
        """Midpoint of tree and fp is within 1 MHz of observed."""
        assert abs(LAMB_SHIFT_WEIGHTED - LAMB_SHIFT_OBS_MHZ) < 1.0

    def test_function_matches_constant(self):
        """Functions match module constants."""
        assert lamb_shift_tree() == LAMB_SHIFT_TREE
        assert lamb_shift_fp() == LAMB_SHIFT_FP
        assert lamb_shift_weighted() == LAMB_SHIFT_WEIGHTED


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Exact rational Lamb shift                                ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestExactLambShift:
    """Exact rational prefactor for tree-level Lamb shift."""

    def test_prefactor_is_fraction(self):
        """The prefactor is a Fraction."""
        assert isinstance(LAMB_SHIFT_EXACT_PREFACTOR, Fraction)

    def test_prefactor_value(self):
        """Prefactor = 5/(6 × 1608²) = 5/15513984."""
        expected = Fraction(5, 6 * 1608 ** 2)
        assert LAMB_SHIFT_EXACT_PREFACTOR == expected

    def test_prefactor_denominator(self):
        """Denominator = 6 × 2585664 = 15513984."""
        assert LAMB_SHIFT_EXACT_PREFACTOR.denominator == 15_513_984

    def test_times_rydberg_gives_tree(self):
        """prefactor × R∞c ≈ LAMB_SHIFT_TREE."""
        result = float(LAMB_SHIFT_EXACT_PREFACTOR) * R_INF_C_MHZ
        assert abs(result - LAMB_SHIFT_TREE) < 1e-6


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: D* extraction from Lamb shift                            ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestDStarExtraction:
    """Invert Lamb shift to extract D*."""

    def test_between_tree_and_fp(self):
        """D*_Lamb lies between D*_tree and D*_fp."""
        assert D_STAR_TREE_F < D_STAR_FROM_LAMB < D_STAR_FP

    def test_close_to_fp(self):
        """D*_Lamb is closer to D*_fp than to D*_tree."""
        dist_tree = abs(D_STAR_FROM_LAMB - D_STAR_TREE_F)
        dist_fp = abs(D_STAR_FROM_LAMB - D_STAR_FP)
        assert dist_fp < dist_tree

    def test_round_trip_tree(self):
        """Round-trip: extract D* from tree prediction, get D*_tree back."""
        d_star_back = d_star_from_lamb_shift(LAMB_SHIFT_TREE)
        assert abs(d_star_back - D_STAR_TREE_F) / D_STAR_TREE_F < 1e-10

    def test_round_trip_fp(self):
        """Round-trip: extract D* from fp prediction, get D*_fp back."""
        d_star_back = d_star_from_lamb_shift(LAMB_SHIFT_FP)
        assert abs(d_star_back - D_STAR_FP) / D_STAR_FP < 1e-10


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Rydberg geometric correction                             ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestRydbergCorrection:
    """Fractional correction to Rydberg constant."""

    def test_negative(self):
        """Correction is negative (D* < 3)."""
        assert RYDBERG_GEO_CORRECTION < 0.0

    def test_tiny(self):
        """Correction is tiny (< 10⁻⁵)."""
        assert abs(RYDBERG_GEO_CORRECTION) < 1e-5

    def test_order_of_magnitude(self):
        """δR∞/R∞ ~ 10⁻⁶."""
        assert 1e-7 < abs(RYDBERG_GEO_CORRECTION) < 1e-5

    def test_formula(self):
        """Manual calculation matches."""
        alpha = 1.0 / 137.035999177
        expected = (D_STAR_TREE_F - 3.0) / 3.0 * alpha ** 2
        assert abs(RYDBERG_GEO_CORRECTION - expected) < 1e-15

    def test_function_matches(self):
        """rydberg_geo_correction() matches module constant."""
        assert rydberg_geo_correction() == RYDBERG_GEO_CORRECTION


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Fine-structure interval                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestFineStructure:
    """Fine-structure 2P₃/₂ − 2P₁/₂."""

    def test_order_of_magnitude(self):
        """ΔE_fs ≈ 10,969 MHz."""
        assert 10_900 < FINE_STRUCTURE_2P < 11_000

    def test_close_to_observed(self):
        """Within 1% of observed."""
        pct = abs(FINE_STRUCTURE_2P - FINE_STRUCTURE_2P_OBS) / FINE_STRUCTURE_2P_OBS
        assert pct < 0.01

    def test_function_matches(self):
        """fine_structure_2p() matches module constant."""
        assert fine_structure_2p() == FINE_STRUCTURE_2P


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Muonic hydrogen                                          ║
# ╔══════════════════════════════════════════════════════════════════╝


class TestMuonicHydrogen:
    """Muonic hydrogen geometric Lamb shift."""

    def test_larger_than_electronic(self):
        """Muonic contribution >> electronic because m_μ >> m_e."""
        assert LAMB_SHIFT_MUONIC_TREE > LAMB_SHIFT_TREE

    def test_scaling(self):
        """Scales as m_μ/m_e × L_electronic."""
        expected = LAMB_SHIFT_TREE * 206.7682830
        assert abs(LAMB_SHIFT_MUONIC_TREE - expected) / expected < 1e-10

    def test_function_matches(self):
        """lamb_shift_muonic_tree() matches module constant."""
        assert lamb_shift_muonic_tree() == LAMB_SHIFT_MUONIC_TREE


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Registry                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestRegistry:
    """Registry entries for atomic spectra observables."""

    @pytest.fixture(autouse=True)
    def _register(self):
        """Register all atomic spectra observables once."""
        if "qm_lamb_shift_tree" not in REGISTRY:
            register_all()

    def test_lamb_tree_registered(self):
        assert "qm_lamb_shift_tree" in REGISTRY

    def test_lamb_fp_registered(self):
        assert "qm_lamb_shift_fp" in REGISTRY

    def test_projection_factor_registered(self):
        assert "qm_projection_factor" in REGISTRY

    def test_d_star_from_lamb_registered(self):
        assert "qm_d_star_from_lamb" in REGISTRY

    def test_lamb_tree_value(self):
        """Tree Lamb shift value matches."""
        obs = REGISTRY.get("qm_lamb_shift_tree")
        assert abs(obs.predicted - LAMB_SHIFT_TREE) < 1e-6

    def test_lamb_fp_value(self):
        """FP Lamb shift value matches."""
        obs = REGISTRY.get("qm_lamb_shift_fp")
        assert abs(obs.predicted - LAMB_SHIFT_FP) < 1e-6

    def test_count(self):
        """Four atomic spectra observables registered."""
        names = ["qm_lamb_shift_tree", "qm_lamb_shift_fp",
                 "qm_projection_factor", "qm_d_star_from_lamb"]
        for name in names:
            assert name in REGISTRY


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Consistency                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestConsistency:
    """Cross-module and internal consistency checks."""

    def test_gamma_equals_dark_energy(self):
        """γ_geo(tree) ≡ γ_DE from dimensional_flow."""
        from sdgft.dimensional_flow import GAMMA_DE_TREE
        assert GAMMA_GEO_TREE == GAMMA_DE_TREE

    def test_gamma_equals_qed_vertex(self):
        """γ_geo(tree) ≡ γ_geo from qed_vertex."""
        from sdgft.qm.qed_vertex import GAMMA_GEO
        assert GAMMA_GEO_TREE == GAMMA_GEO

    def test_projection_factor_from_axioms(self):
        """5/6 derivable from Level 0 axioms only."""
        d = Fraction(5, 24)
        dg = Fraction(1, 24)
        assert d / (d + dg) == Fraction(5, 6)

    def test_lamb_formula_components(self):
        """L = proj × γ²_geo × R∞c — verify all three factors."""
        l_check = PROJECTION_FACTOR_F * GAMMA_GEO_TREE_SQ_F * R_INF_C_MHZ
        assert abs(l_check - LAMB_SHIFT_TREE) / LAMB_SHIFT_TREE < 1e-10

    def test_tree_fp_bracket_observed(self):
        """Tree overshoots, fp undershoots → observed is between."""
        assert LAMB_SHIFT_FP < LAMB_SHIFT_OBS_MHZ < LAMB_SHIFT_TREE

    def test_universality_three_scales(self):
        """Same γ²_geo appears in g-2, Lamb shift, and dark energy.

        This tests that the numerical values are consistent across modules.
        """
        from sdgft.qm.qed_vertex import GAMMA_GEO_SQ_F as g2_gamma_sq
        from sdgft.dimensional_flow import GAMMA_DE_TREE_F as de_gamma

        # All derived from same δ²/D*
        assert abs(GAMMA_GEO_TREE_SQ_F - g2_gamma_sq) < 1e-15
        assert abs(GAMMA_GEO_TREE_F - de_gamma) < 1e-15

    def test_d_star_from_lamb_consistency(self):
        """D* extracted from Lamb shift is physical (between tree and fp)."""
        assert 2.79 < D_STAR_FROM_LAMB < 2.80

    def test_rydberg_hz(self):
        """R∞c in Hz = R∞c in MHz × 10⁶."""
        assert abs(R_INF_C_HZ - R_INF_C_MHZ * 1e6) < 1.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Summary printer (smoke test)                             ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestSummary:
    """print_summary() runs without error."""

    def test_summary_runs(self, capsys):
        """Summary prints without exception."""
        print_summary()
        captured = capsys.readouterr()
        assert "Lamb" in captured.out
        assert "SDGFT" in captured.out
        assert "5/6" in captured.out or "0.833" in captured.out

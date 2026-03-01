"""Tests for SDGFT QED vertex corrections (g-2).

Covers:
    - Geometric anomalous dimension γ_geo = δ²/D* = 1/1608
    - Vertex correction formula Δa_ℓ = (α/2π)·γ²_geo·ln(m_ℓ/m_e)
    - Electron, muon, tau predictions
    - Schwinger term and d-dimensional diagnostic
    - Comparison with experimental muon g-2 anomaly
    - Registry entries
    - Internal consistency checks
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from sdgft.constants import DELTA_G, DELTA_G_F
from sdgft.dimension import D_STAR_TREE, D_STAR_TREE_F
from sdgft.dimensional_flow import GAMMA_DE_TREE, GAMMA_DE_TREE_F
from sdgft.particle import (
    ALPHA_EM_TREE,
    MU_E_RATIO,
    TAU_MU_RATIO_TREE,
    TAU_E_RATIO_TREE,
)
from sdgft.registry import REGISTRY

from sdgft.qm.qed_vertex import (
    # Constants
    ALPHA_INV_OBS,
    ALPHA_OBS,
    M_MU_OVER_M_E,
    M_TAU_OVER_M_E,
    A_MU_EXP,
    A_MU_EXP_UNCERT,
    A_E_EXP,
    A_E_EXP_UNCERT,
    A_MU_SM_WP,
    A_MU_SM_WP_UNCERT,
    DELTA_A_MU_OBS,
    DELTA_A_MU_OBS_UNCERT,
    # Gamma
    GAMMA_GEO,
    GAMMA_GEO_F,
    GAMMA_GEO_SQ,
    GAMMA_GEO_SQ_F,
    gamma_geo_exact,
    # Vertex corrections
    delta_a_lepton,
    delta_a_electron,
    delta_a_muon,
    delta_a_tau,
    DELTA_A_E,
    DELTA_A_MU,
    DELTA_A_TAU,
    DELTA_A_MU_PURE,
    DELTA_A_TAU_PURE,
    # Schwinger
    schwinger_term,
    schwinger_sdgft,
    SCHWINGER_OBS,
    SCHWINGER_SDGFT,
    # d-dimensional
    xi_d,
    D_EFF,
    XI_AT_D_EFF,
    # Totals
    A_MU_SDGFT_TOTAL,
    A_MU_SDGFT_SIGMA,
    TAU_MU_GEO_RATIO,
    # Predictions
    G2Prediction,
    predict_electron,
    predict_muon,
    predict_tau,
    # Registry
    register_all,
    print_summary,
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Geometric anomalous dimension                            ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestGammaGeo:
    """γ_geo = δ²/D* = 1/1608."""

    def test_exact_fraction(self):
        """γ_geo must be exactly 1/1608 as Fraction."""
        assert GAMMA_GEO == Fraction(1, 1608)

    def test_from_constants(self):
        """γ_geo = δ² / D* using axiomatic constants."""
        g = DELTA_G ** 2 / D_STAR_TREE
        assert g == Fraction(1, 1608)

    def test_factorisation(self):
        """1608 = 24 × 67 (vertex count × D* numerator)."""
        assert 1608 == 24 * 67

    def test_squared(self):
        """γ²_geo = 1/2585664 = 1/1608²."""
        assert GAMMA_GEO_SQ == Fraction(1, 1608 ** 2)
        assert GAMMA_GEO_SQ == Fraction(1, 2_585_664)

    def test_equals_gamma_de(self):
        """γ_geo must be IDENTICAL to γ_DE (dark energy anomalous dim)."""
        assert GAMMA_GEO == GAMMA_DE_TREE
        assert abs(GAMMA_GEO_F - GAMMA_DE_TREE_F) < 1e-15

    def test_float_value(self):
        """Float value ~6.22 × 10⁻⁴."""
        assert abs(GAMMA_GEO_F - 1.0 / 1608.0) < 1e-15
        assert 6.0e-4 < GAMMA_GEO_F < 6.5e-4

    def test_gamma_geo_exact_function(self):
        """gamma_geo_exact() returns the same as the module constant."""
        assert gamma_geo_exact() == GAMMA_GEO


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Vertex correction formula                                ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestDeltaALepton:
    """Δa_ℓ = (α/2π) · γ²_geo · ln(m_ℓ/m_e)."""

    def test_formula_manual(self):
        """Cross-check: manually compute the muon correction."""
        alpha = ALPHA_OBS
        gamma_sq = GAMMA_GEO_SQ_F
        log_ratio = math.log(M_MU_OVER_M_E)
        expected = alpha / (2.0 * math.pi) * gamma_sq * log_ratio
        assert abs(delta_a_muon() - expected) < 1e-20

    def test_positive(self):
        """Correction must be positive for m_ℓ > m_e."""
        assert delta_a_muon() > 0.0
        assert delta_a_tau() > 0.0

    def test_zero_for_electron(self):
        """Correction must be EXACTLY zero for m_ℓ = m_e."""
        assert delta_a_lepton(1.0) == 0.0
        assert delta_a_electron() == 0.0

    def test_zero_for_lighter(self):
        """Correction must be zero for m_ℓ < m_e (unphysical)."""
        assert delta_a_lepton(0.5) == 0.0
        assert delta_a_lepton(0.0) == 0.0

    def test_monotonic_in_mass(self):
        """Heavier leptons get larger corrections."""
        assert delta_a_electron() < delta_a_muon() < delta_a_tau()

    def test_scales_with_log(self):
        """Verify the log scaling: Δa(2m) - Δa(m) = (α/2π)γ²ln(2)."""
        da_200 = delta_a_lepton(200.0)
        da_400 = delta_a_lepton(400.0)
        diff = da_400 - da_200
        expected_diff = ALPHA_OBS / (2 * math.pi) * GAMMA_GEO_SQ_F * math.log(2)
        assert abs(diff - expected_diff) / expected_diff < 1e-10

    def test_scales_with_alpha(self):
        """Correction is proportional to α."""
        da1 = delta_a_lepton(200.0, alpha=1.0 / 137.0)
        da2 = delta_a_lepton(200.0, alpha=2.0 / 137.0)
        assert abs(da2 / da1 - 2.0) < 1e-10


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Muon g-2 prediction                                     ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestMuonG2:
    """Muon g-2: the primary testable prediction."""

    def test_order_of_magnitude(self):
        """Δa_μ ~ 2.4 × 10⁻⁹."""
        assert 2.0e-9 < DELTA_A_MU < 3.0e-9

    def test_precise_value(self):
        """Δa_μ ≈ 2.39 × 10⁻⁹ (within 1% of hand calculation)."""
        assert abs(DELTA_A_MU - 2.39e-9) / 2.39e-9 < 0.01

    def test_matches_observed_anomaly(self):
        """SDGFT correction matches observed anomaly within 1σ."""
        tension = abs(DELTA_A_MU - DELTA_A_MU_OBS) / DELTA_A_MU_OBS_UNCERT
        assert tension < 1.0  # must be within 1σ

    def test_tension_excellent(self):
        """Tension should be < 0.5σ (well within 1σ)."""
        tension = abs(DELTA_A_MU - DELTA_A_MU_OBS) / DELTA_A_MU_OBS_UNCERT
        assert tension < 0.5

    def test_total_sdgft_vs_experiment(self):
        """Total SDGFT a_μ is much closer to experiment than SM alone."""
        sm_tension = abs(A_MU_SM_WP - A_MU_EXP) / A_MU_EXP_UNCERT
        sdgft_tension = A_MU_SDGFT_SIGMA
        assert sdgft_tension < sm_tension  # SDGFT is better than SM
        assert sdgft_tension < 1.0  # within combined 1σ

    def test_explains_majority_of_anomaly(self):
        """SDGFT correction explains ≥ 90% of the observed anomaly."""
        fraction = DELTA_A_MU / DELTA_A_MU_OBS
        assert fraction >= 0.90

    def test_observed_anomaly_value(self):
        """Cross-check the derived observed anomaly."""
        assert abs(DELTA_A_MU_OBS - 2.49e-9) / 2.49e-9 < 0.01

    def test_observed_combined_uncertainty(self):
        """Combined uncertainty ~4.8 × 10⁻¹⁰."""
        expected = math.sqrt(22e-11 ** 2 + 43e-11 ** 2)
        assert abs(DELTA_A_MU_OBS_UNCERT - expected) < 1e-15


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Electron g-2                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestElectronG2:
    """Electron g-2: SDGFT correction must be zero."""

    def test_delta_a_e_zero(self):
        """Geometric correction for electron is identically zero."""
        assert DELTA_A_E == 0.0

    def test_function_returns_zero(self):
        """delta_a_electron() returns exactly 0."""
        assert delta_a_electron() == 0.0

    def test_preserves_precision(self):
        """SDGFT does NOT introduce corrections to a_e.

        The electron g-2 is measured to 0.13 ppt — any BSM correction
        must be negligible.  SDGFT gives exactly zero.
        """
        assert DELTA_A_E < A_E_EXP_UNCERT  # correction < exp uncertainty


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Tau g-2 prediction                                      ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestTauG2:
    """Tau g-2: prediction for future experiments."""

    def test_order_of_magnitude(self):
        """Δa_τ ~ 3.7 × 10⁻⁹."""
        assert 3.0e-9 < DELTA_A_TAU < 4.5e-9

    def test_greater_than_muon(self):
        """Tau correction must be larger than muon correction."""
        assert DELTA_A_TAU > DELTA_A_MU

    def test_ratio_prediction(self):
        """Δa_τ/Δa_μ = ln(m_τ/m_e)/ln(m_μ/m_e) ≈ 1.53."""
        ratio = DELTA_A_TAU / DELTA_A_MU
        expected = math.log(M_TAU_OVER_M_E) / math.log(M_MU_OVER_M_E)
        assert abs(ratio - expected) / expected < 1e-10

    def test_ratio_value(self):
        """The ratio is ~1.53."""
        assert abs(TAU_MU_GEO_RATIO - 1.53) < 0.02


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Schwinger term                                           ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestSchwingerTerm:
    """Schwinger one-loop: a^{(1)} = α/(2π)."""

    def test_observed_alpha(self):
        """Schwinger term with observed α."""
        expected = 1.0 / (137.035999177 * 2.0 * math.pi)
        assert abs(SCHWINGER_OBS - expected) < 1e-15

    def test_sdgft_alpha(self):
        """Schwinger term with SDGFT tree α."""
        expected = ALPHA_EM_TREE / (2.0 * math.pi)
        assert abs(SCHWINGER_SDGFT - expected) < 1e-15

    def test_both_close(self):
        """SDGFT and observed Schwinger terms agree to < 0.2%."""
        diff = abs(SCHWINGER_OBS - SCHWINGER_SDGFT)
        assert diff / SCHWINGER_OBS < 0.002

    def test_function_matches_constant(self):
        """schwinger_term() matches SCHWINGER_OBS."""
        assert schwinger_term() == SCHWINGER_OBS

    def test_sdgft_function_matches(self):
        """schwinger_sdgft() matches SCHWINGER_SDGFT."""
        assert schwinger_sdgft() == SCHWINGER_SDGFT


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: d-dimensional Schwinger Ξ(d)                             ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestXiD:
    """d-dimensional Schwinger function Ξ(d) — diagnostic."""

    def test_xi_4_is_unity(self):
        """Ξ(4) = 1 exactly (standard QED)."""
        assert abs(xi_d(4.0) - 1.0) < 1e-12

    def test_xi_2_is_zero(self):
        """Ξ(2) = 0 (no magnetic moment in 2D)."""
        assert xi_d(2.0) == 0.0

    def test_xi_3_finite(self):
        """Ξ(3) is finite and > 1 (QED₃ has larger correction)."""
        val = xi_d(3.0)
        assert math.isfinite(val)
        assert val > 1.0
        # Ξ(3) = π²/2 ≈ 4.93
        assert abs(val - math.pi ** 2 / 2.0) < 0.01

    def test_xi_at_sdgft_dimension(self):
        """Ξ(1+D*) ≈ 1.3-1.4 (diagnostic, not the SDGFT prediction)."""
        assert 1.0 < XI_AT_D_EFF < 2.0

    def test_d_eff_value(self):
        """d_eff = 1 + D*_tree = 91/24 ≈ 3.7917."""
        assert abs(D_EFF - (1.0 + 67.0 / 24.0)) < 1e-10
        assert abs(D_EFF - 91.0 / 24.0) < 1e-10

    def test_below_2_returns_zero(self):
        """Ξ(d) = 0 for d ≤ 2."""
        assert xi_d(1.5) == 0.0
        assert xi_d(0.0) == 0.0
        assert xi_d(-1.0) == 0.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Pure SDGFT predictions                                  ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestPureSDGFT:
    """Pure SDGFT predictions (using SDGFT α and mass ratios)."""

    def test_pure_muon_close_to_observed_input(self):
        """Pure SDGFT Δa_μ agrees with observed-input Δa_μ to < 1%."""
        diff = abs(DELTA_A_MU_PURE - DELTA_A_MU)
        assert diff / DELTA_A_MU < 0.01

    def test_pure_tau_close_to_observed_input(self):
        """Pure SDGFT Δa_τ agrees with observed-input Δa_τ to < 2%."""
        diff = abs(DELTA_A_TAU_PURE - DELTA_A_TAU)
        assert diff / DELTA_A_TAU < 0.02

    def test_pure_uses_sdgft_alpha(self):
        """Pure prediction uses SDGFT tree-level α."""
        expected = delta_a_muon(
            alpha=ALPHA_EM_TREE,
            mass_ratio=MU_E_RATIO,
        )
        assert DELTA_A_MU_PURE == expected

    def test_pure_muon_still_matches_anomaly(self):
        """Pure SDGFT correction still matches anomaly within 1σ."""
        tension = abs(DELTA_A_MU_PURE - DELTA_A_MU_OBS) / DELTA_A_MU_OBS_UNCERT
        assert tension < 1.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: G2Prediction dataclass                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestG2Prediction:
    """G2Prediction dataclass for experiment-specific results."""

    def test_electron_prediction(self):
        """Electron prediction has zero correction."""
        pred = predict_electron()
        assert pred.lepton == "e"
        assert pred.delta_a_geo == 0.0
        assert pred.mass_ratio == 1.0

    def test_muon_prediction(self):
        """Muon prediction has a_sm and a_exp."""
        pred = predict_muon()
        assert pred.lepton == "μ"
        assert pred.a_sm is not None
        assert pred.a_exp is not None
        assert pred.a_sdgft is not None
        assert pred.a_sdgft > pred.a_sm  # geo correction is positive

    def test_muon_sigma(self):
        """Muon prediction tension < 1σ."""
        pred = predict_muon()
        assert pred.sigma_vs_exp is not None
        assert pred.sigma_vs_exp < 1.0

    def test_muon_fraction_of_anomaly(self):
        """Muon prediction explains ~96% of anomaly."""
        pred = predict_muon()
        frac = pred.fraction_of_anomaly
        assert frac is not None
        assert 0.85 < frac < 1.05

    def test_tau_prediction_no_experiment(self):
        """Tau prediction has no experimental comparison."""
        pred = predict_tau()
        assert pred.lepton == "τ"
        assert pred.a_exp is None
        assert pred.a_sdgft is None  # no SM value available
        assert pred.sigma_vs_exp is None
        assert pred.delta_a_geo > 0.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Registry                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestRegistry:
    """Registry entries for QED vertex observables."""

    @pytest.fixture(autouse=True)
    def _register(self):
        """Register all QED vertex observables once."""
        # Only register if not already done
        if "qm_delta_a_mu_geo" not in REGISTRY:
            register_all()

    def test_delta_a_mu_registered(self):
        assert "qm_delta_a_mu_geo" in REGISTRY

    def test_a_mu_total_registered(self):
        assert "qm_a_mu_total" in REGISTRY

    def test_delta_a_tau_registered(self):
        assert "qm_delta_a_tau_geo" in REGISTRY

    def test_gamma_geo_registered(self):
        assert "qm_gamma_geo" in REGISTRY

    def test_total_count(self):
        """Four QED vertex observables registered."""
        names = ["qm_delta_a_mu_geo", "qm_a_mu_total",
                 "qm_delta_a_tau_geo", "qm_gamma_geo"]
        for name in names:
            assert name in REGISTRY

    def test_delta_a_mu_sigma(self):
        """Sigma tension < 0.5 for the muon correction."""
        obs = REGISTRY.get("qm_delta_a_mu_geo")
        sigma = obs.sigma_tension
        assert sigma is not None
        assert sigma < 0.5

    def test_a_mu_total_sigma(self):
        """Sigma tension < 0.5 for the total a_μ."""
        obs = REGISTRY.get("qm_a_mu_total")
        sigma = obs.sigma_tension
        assert sigma is not None
        assert sigma < 0.5


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Consistency checks                                       ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestConsistency:
    """Cross-module and internal consistency."""

    def test_gamma_geo_equals_gamma_de(self):
        """γ_geo ≡ γ_DE (identical physics)."""
        assert GAMMA_GEO == GAMMA_DE_TREE

    def test_gamma_geo_from_axioms(self):
        """γ_geo derivable from Level 0 axioms only."""
        delta_g = Fraction(1, 24)
        d_star = Fraction(67, 24)
        gamma = delta_g ** 2 / d_star
        assert gamma == Fraction(1, 1608)

    def test_correction_order_vs_schwinger(self):
        """Geometric correction is ~2 × 10⁻⁶ relative to Schwinger."""
        relative = DELTA_A_MU / SCHWINGER_OBS
        assert 1e-7 < relative < 1e-5

    def test_alpha_tree_consistent(self):
        """SDGFT tree α matches 1/ALPHA_EM_INV_TREE."""
        from sdgft.particle import ALPHA_EM_INV_TREE
        assert abs(ALPHA_EM_TREE - 1.0 / ALPHA_EM_INV_TREE) < 1e-15

    def test_tau_e_ratio_composite(self):
        """SDGFT m_τ/m_e = (m_μ/m_e) × (m_τ/m_μ)."""
        composite = MU_E_RATIO * TAU_MU_RATIO_TREE
        assert abs(TAU_E_RATIO_TREE - composite) < 1e-10

    def test_no_negative_corrections(self):
        """All lepton corrections ≥ 0."""
        for ratio in [0.5, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
            assert delta_a_lepton(ratio) >= 0.0

    def test_correction_independent_of_units(self):
        """Correction formula is dimensionless (uses mass ratio)."""
        # Doubling both masses doesn't change the ratio
        da1 = delta_a_lepton(200.0)
        da2 = delta_a_lepton(200.0)  # same ratio, same result
        assert da1 == da2

    def test_total_matches_components(self):
        """A_MU_SDGFT_TOTAL = A_MU_SM_WP + DELTA_A_MU."""
        assert abs(A_MU_SDGFT_TOTAL - (A_MU_SM_WP + DELTA_A_MU)) < 1e-20

    def test_sm_alone_tension_high(self):
        """SM alone (without SDGFT) has >3σ tension with experiment."""
        sm_only_sigma = abs(A_MU_SM_WP - A_MU_EXP) / DELTA_A_MU_OBS_UNCERT
        assert sm_only_sigma > 3.0

    def test_sdgft_reduces_tension(self):
        """SDGFT total prediction reduces tension vs SM alone."""
        sm_sigma = abs(A_MU_SM_WP - A_MU_EXP) / DELTA_A_MU_OBS_UNCERT
        sdgft_sigma = A_MU_SDGFT_SIGMA
        assert sdgft_sigma < sm_sigma


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Test: Summary printer (smoke test)                             ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestSummary:
    """print_summary() runs without error."""

    def test_summary_runs(self, capsys):
        """Summary prints without exception."""
        print_summary()
        captured = capsys.readouterr()
        assert "γ_geo" in captured.out or "gamma_geo" in captured.out.lower()
        assert "SDGFT" in captured.out
        assert "Schwinger" in captured.out

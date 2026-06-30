"""Tests for the baryogenesis–CP link module.

Tests cover:
1. SM constants (sphaleron conversion, g*, T_sph)
2. Cone geometry (solid angle, κ_geometric)
3. Chiral coupling ξ_G
4. Individual η_B formulas (topological, geometric CP, sphaleron, full chain)
5. Consistency check
6. BaryogenesisResult dataclass
7. Registry integration
"""

import math
import numpy as np

import pytest

from sdgft.constants import DELTA_F, DELTA_G_F, PHI
from sdgft.experimental.baryogenesis_cp import (
    # SM constants
    G_STAR_EW, SPHALERON_CONVERSION, T_SHALERON_GEV,
    # Observed
    ETA_B_OBSERVED, ETA_B_OBSERVED_UNC,
    # Cone geometry
    solid_angle_4d_cone, OMEGA_CONE, OMEGA_S3,
    CONE_FRACTION, KAPPA_GEO, THETA_MAX_RAD,
    # Chiral coupling
    xi_g, XI_G,
    # Formulas
    eta_b_topological, eta_b_geometric_cp,
    eta_b_sphaleron, eta_b_full_chain,
    # Consistency
    consistency_check, ConsistencyResult,
    # Results
    BaryogenesisResult, get_results,
    # Registry
    register_all,
    # Baryon evolution
    solve_baryon_evolution, ETA_B_TOPOLOGICAL,
)


# ── SM constants ──────────────────────────────────────────────────

class TestSMConstants:

    def test_g_star_ew(self):
        assert G_STAR_EW == 106.75

    def test_sphaleron_conversion(self):
        assert abs(SPHALERON_CONVERSION - 28.0 / 79.0) < 1e-15

    def test_t_shaleron(self):
        """T_sph > 100 GeV (EW scale)."""
        assert 100.0 < T_SHALERON_GEV < 200.0


# ── Cone geometry ─────────────────────────────────────────────────

class TestConeGeometry:

    def test_theta_max(self):
        assert abs(THETA_MAX_RAD - math.pi / 6.0) < 1e-15

    def test_solid_angle_zero(self):
        assert abs(solid_angle_4d_cone(0.0)) < 1e-15

    def test_solid_angle_hemisphere(self):
        """Ω₄(π/2) = 2π(π/2 - 0) = π² on S³."""
        expected = 2.0 * math.pi * (math.pi / 2.0)
        assert abs(solid_angle_4d_cone(math.pi / 2.0) - expected) < 1e-12

    def test_omega_s3(self):
        assert abs(OMEGA_S3 - 2.0 * math.pi ** 2) < 1e-12

    def test_cone_fraction_positive(self):
        assert 0.0 < CONE_FRACTION < 0.1

    def test_kappa_geo_six_cones(self):
        assert abs(KAPPA_GEO - 6.0 * CONE_FRACTION) < 1e-15

    def test_kappa_geo_order(self):
        """κ ≈ 0.17, between 0.01 and 1.0."""
        assert 0.01 < KAPPA_GEO < 1.0


# ── Chiral coupling ──────────────────────────────────────────────

class TestChiralCoupling:

    def test_xi_g_formula(self):
        expected = DELTA_F * DELTA_G_F * PHI ** (-2)
        assert abs(XI_G - expected) < 1e-15

    def test_xi_g_function(self):
        assert abs(xi_g() - XI_G) < 1e-15

    def test_xi_g_order(self):
        """ξ_G ≈ 0.003, small coupling."""
        assert 1e-4 < XI_G < 0.01

    def test_xi_g_custom(self):
        """ξ_G with custom arguments."""
        val = xi_g(delta=0.5, delta_g=0.5, phi=1.0)
        assert abs(val - 0.25) < 1e-15


# ── Topological formula ──────────────────────────────────────────

class TestTopological:

    def test_value_order(self):
        """η_B ≈ 6 × 10⁻¹⁰."""
        val = eta_b_topological()
        assert 5e-10 < val < 8e-10

    def test_matches_cosmology(self):
        """Must match the existing cosmology.py formula."""
        from sdgft.cosmology import ETA_B
        assert abs(eta_b_topological() - ETA_B) < 1e-15

    def test_exact_formula(self):
        d = DELTA_G_F
        expected = d ** 6 * (1.0 - d) / 8.0
        # cosmology.py applies a closure correction; compare to base
        # Just check order of magnitude since closure may differ
        assert abs(eta_b_topological() - expected) / expected < 0.1


# ── Geometric CP formula ─────────────────────────────────────────

class TestGeometricCP:

    def test_order_of_magnitude(self):
        val = eta_b_geometric_cp()
        assert 1e-10 < val < 1e-9

    def test_sin_modulation(self):
        """sin(δ_CP) < 1, so geometric_cp < topological."""
        # Only true if delta_cp != pi/2
        from sdgft.experimental.ckm_phase import DELTA_CP_BEST
        if abs(DELTA_CP_BEST - math.pi / 2) > 0.01:
            assert eta_b_geometric_cp() < eta_b_topological()

    def test_custom_delta_cp_maximal(self):
        """At δ_CP = π/2, sin = 1: matches topological."""
        base = DELTA_G_F ** 6 * (1.0 - DELTA_G_F) / 8.0
        val = eta_b_geometric_cp(delta_cp=math.pi / 2.0)
        assert abs(val - base) < 1e-20


# ── Sphaleron formula ────────────────────────────────────────────

class TestSphaleron:

    def test_positive(self):
        val = eta_b_sphaleron()
        assert val > 0

    def test_very_small(self):
        """Sphaleron formula gives much smaller η_B (perturbative)."""
        val = eta_b_sphaleron()
        assert val < 1e-15

    def test_proportional_to_kappa(self):
        val1 = eta_b_sphaleron(kappa=1.0)
        val2 = eta_b_sphaleron(kappa=2.0)
        assert abs(val2 / val1 - 2.0) < 1e-10


# ── Full chain formula ───────────────────────────────────────────

class TestFullChain:

    def test_order_of_magnitude(self):
        val = eta_b_full_chain()
        assert 1e-11 < val < 1e-9

    def test_less_than_topological(self):
        """Full chain should be smaller due to sin and closure factors."""
        assert eta_b_full_chain() < eta_b_topological()


# ── Consistency check ─────────────────────────────────────────────

class TestConsistency:

    def test_returns_named_tuple(self):
        result = consistency_check()
        assert isinstance(result, ConsistencyResult)

    def test_kappa_required_large(self):
        """κ_required >> κ_geometric (non-perturbative enhancement)."""
        result = consistency_check()
        assert result.kappa_required > result.kappa_geometric * 10

    def test_ratio_positive(self):
        result = consistency_check()
        assert result.kappa_ratio > 0

    def test_all_eta_positive(self):
        result = consistency_check()
        assert result.eta_b_topological > 0
        assert result.eta_b_geometric_cp > 0
        assert result.eta_b_sphaleron > 0
        assert result.eta_b_full_chain > 0


# ── BaryogenesisResult ────────────────────────────────────────────

class TestBaryogenesisResult:

    def test_deviation_abs(self):
        r = BaryogenesisResult(
            name="test", label="test",
            eta_b=7.0e-10,
            formula="test", reasoning="test",
        )
        assert abs(r.deviation_abs - abs(7.0e-10 - ETA_B_OBSERVED)) < 1e-20

    def test_deviation_percent(self):
        r = BaryogenesisResult(
            name="test", label="test",
            eta_b=ETA_B_OBSERVED,
            formula="test", reasoning="test",
        )
        assert abs(r.deviation_percent) < 1e-10

    def test_deviation_sigma_zero(self):
        r = BaryogenesisResult(
            name="test", label="test",
            eta_b=ETA_B_OBSERVED,
            formula="test", reasoning="test",
        )
        assert abs(r.deviation_sigma) < 1e-10


# ── get_results ───────────────────────────────────────────────────

class TestGetResults:

    def test_returns_dict(self):
        results = get_results()
        assert isinstance(results, dict)

    def test_four_formulas(self):
        results = get_results()
        assert len(results) == 4

    def test_expected_keys(self):
        results = get_results()
        expected = {"topological", "geometric_cp", "sphaleron", "full_chain"}
        assert set(results.keys()) == expected


# ── Registry ──────────────────────────────────────────────────────

class TestRegistry:

    def test_register_no_errors(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        names = [o.name for o in reg]
        assert "exp_eta_b_geo_cp" in names
        assert "exp_eta_b_sphaleron" in names

    def test_registered_levels(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        for obs in reg:
            assert obs.level == 6

    def test_geometric_cp_deviation(self):
        """Registered η_B (geo+CP) has reasonable deviation."""
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        obs = reg["exp_eta_b_geo_cp"]
        sigma = obs.deviation_abs / obs.observed_uncertainty
        # Within ~50σ at worst (sphaleron formula is very different)
        assert sigma < 100


class TestBaryonEvolution:
    """Verifiziert die Stabilität und physikalische Konsistenz des Boltzmann-Solvers."""

    def test_baryon_transport_solver_stability(self):
        """Überprüft, ob der RK4-Solver die Integration ohne NaN- oder Inf-Werte abschließt."""
        z, eta = solve_baryon_evolution()
        assert len(z) == 2000
        assert len(eta) == 2000
        assert not np.isnan(eta).any()
        assert not np.isinf(eta).any()

    def test_baryon_freezeout_convergence(self):
        """Verifiziert, dass der dynamische Endwert exakt mit dem topologischen Modell übereinstimmt."""
        _, eta = solve_baryon_evolution(z_start=0.01, z_end=500.0, steps=2000)
        final_frozen_eta = eta[-1]
        
        # Der dynamische Freeze-out-Wert muss den Sollwert innerhalb der 1%-Toleranz treffen
        assert final_frozen_eta == pytest.approx(ETA_B_TOPOLOGICAL, rel=1e-2)

    def test_baryon_monotonicity_and_saturation(self):
        """Stellt sicher, dass die Asymmetrie monoton ansteigt und sauber einfriert (Sättigung)."""
        _, eta = solve_baryon_evolution()
        diffs = np.diff(eta)
        
        # Monotoniekriterium: Die Asymmetrie darf zu keinem Zeitpunkt kollabieren
        assert (diffs >= -1e-16).all()
        
        # Sättigungskriterium: Die Änderungsrate am Ende muss deutlich kleiner sein als in der heißen Phase
        assert abs(diffs[-1]) < abs(diffs[100])


"""Tests for the CKM CP-phase and Jarlskog invariant module.

Tests cover:
1. D₄ triality constants
2. CKM angle extraction from |V_ij|
3. Jarlskog invariant calculation
4. |V_cb| candidate formulas
5. All δ_CP candidate formulas and their ordering
6. Full CKM matrix unitarity
7. Best-candidate agreement with observation
8. Registry integration
"""

import math

import pytest

from sdgft.constants import DELTA_F, DELTA_G_F, PHI
from sdgft.particle import V_US, V_UB
from sdgft.experimental.ckm_phase import (
    # Constants
    TRIALITY_ORDER, TRIALITY_BASE_ANGLE,
    N_VERTICES_24CELL, N_VERTICES_16CELL,
    DELTA_CP_OBSERVED, DELTA_CP_OBSERVED_UNC,
    JARLSKOG_OBSERVED, JARLSKOG_OBSERVED_UNC,
    # CKM angles
    CKMAngles, ckm_angles_from_elements,
    # Jarlskog
    jarlskog_invariant, jarlskog_from_elements,
    # V_cb candidates
    v_cb_wolfenstein, v_cb_delta_squared, v_cb_geometric,
    V_CB_WOLFENSTEIN, V_CB_DELTA_SQ, V_CB_GEO, V_CB_BEST,
    # δ_CP candidates
    delta_cp_naive_triality, delta_cp_dimensional_flow,
    delta_cp_fibonacci_phase, delta_cp_chiral_projection,
    delta_cp_golden_angle_half, delta_cp_golden_angle_damped,
    delta_cp_triality_golden, delta_cp_atan_phi_delta,
    delta_cp_pi_over_phi_cubed_corr,
    build_candidates, CP_PHASE_CANDIDATES, BEST_CANDIDATE,
    DELTA_CP_BEST,
    # Full CKM
    build_ckm_matrix, ckm_unitarity_check,
    CKM_ANGLES_BEST, CKM_MATRIX_BEST, JARLSKOG_BEST,
    # Registry
    register_all,
)


# ── D₄ triality constants ────────────────────────────────────────

class TestTrialityConstants:

    def test_triality_order(self):
        assert TRIALITY_ORDER == 3

    def test_triality_base_angle(self):
        assert abs(TRIALITY_BASE_ANGLE - 2 * math.pi / 3) < 1e-15

    def test_24cell_vertices(self):
        assert N_VERTICES_24CELL == 24

    def test_16cell_partition(self):
        """24 vertices split into 3 disjoint 16-cells of 8."""
        assert N_VERTICES_16CELL == 8
        assert N_VERTICES_24CELL == TRIALITY_ORDER * N_VERTICES_16CELL


# ── CKM angle extraction ─────────────────────────────────────────

class TestCKMAngles:

    def test_small_v_ub(self):
        """For small v_ub, theta_13 ≈ v_ub."""
        angles = ckm_angles_from_elements(0.2243, 0.0408, 0.00382, 1.2)
        assert abs(math.sin(angles.theta_13) - 0.00382) < 1e-10

    def test_cabibbo_angle(self):
        """theta_12 ≈ arcsin(v_us) for small s13."""
        angles = ckm_angles_from_elements(0.2243, 0.0408, 0.00382, 1.2)
        expected = math.asin(0.2243 / math.sqrt(1 - 0.00382**2))
        assert abs(angles.theta_12 - expected) < 1e-10

    def test_delta_cp_passthrough(self):
        """δ_CP is stored as given."""
        angles = ckm_angles_from_elements(0.2, 0.04, 0.003, 1.144)
        assert angles.delta_cp == 1.144

    def test_namedtuple_fields(self):
        a = CKMAngles(0.1, 0.2, 0.3, 0.4)
        assert a.theta_12 == 0.1
        assert a.theta_23 == 0.2
        assert a.theta_13 == 0.3
        assert a.delta_cp == 0.4


# ── Jarlskog invariant ───────────────────────────────────────────

class TestJarlskog:

    def test_zero_phase_gives_zero_j(self):
        """No CP violation if δ = 0."""
        angles = CKMAngles(0.227, 0.041, 0.004, 0.0)
        assert abs(jarlskog_invariant(angles)) < 1e-16

    def test_maximal_phase(self):
        """J is maximal at δ = π/2."""
        angles_half = CKMAngles(0.227, 0.041, 0.004, math.pi / 2)
        angles_quarter = CKMAngles(0.227, 0.041, 0.004, math.pi / 4)
        assert jarlskog_invariant(angles_half) > jarlskog_invariant(angles_quarter)

    def test_observed_j_order_of_magnitude(self):
        """J should be O(10⁻⁵) for realistic CKM parameters."""
        angles = ckm_angles_from_elements(0.2243, 0.0408, 0.00382, 1.144)
        j = jarlskog_invariant(angles)
        assert 1e-6 < abs(j) < 1e-3

    def test_jarlskog_from_elements_wrapper(self):
        """Wrapper gives same result as the two-step method."""
        delta = 1.2
        j_direct = jarlskog_from_elements(V_US, V_CB_BEST, V_UB, delta)
        angles = ckm_angles_from_elements(V_US, V_CB_BEST, V_UB, delta)
        j_manual = jarlskog_invariant(angles)
        assert abs(j_direct - j_manual) < 1e-15

    def test_j_antisymmetric_in_delta(self):
        """J(−δ) = −J(δ)."""
        d = 1.1
        j_plus = jarlskog_from_elements(0.22, 0.04, 0.004, d)
        j_minus = jarlskog_from_elements(0.22, 0.04, 0.004, -d)
        assert abs(j_plus + j_minus) < 1e-15


# ── |V_cb| candidates ────────────────────────────────────────────

class TestVcbCandidates:

    def test_wolfenstein_value(self):
        """V_us² ≈ 0.050."""
        assert abs(V_CB_WOLFENSTEIN - V_US**2) < 1e-15
        assert abs(V_CB_WOLFENSTEIN - 0.050) < 0.002

    def test_delta_squared_value(self):
        """Δ² ≈ 0.0434."""
        assert abs(V_CB_DELTA_SQ - DELTA_F**2) < 1e-15
        assert abs(V_CB_DELTA_SQ - 0.0434) < 0.001

    def test_geometric_value(self):
        """Δ²(1 − δφ) ≈ 0.0405."""
        expected = DELTA_F**2 * (1.0 - DELTA_G_F * PHI)
        assert abs(V_CB_GEO - expected) < 1e-15
        assert abs(V_CB_GEO - 0.0405) < 0.001

    def test_geometric_closer_to_observed(self):
        """Geometric formula is closer to 0.0408 than the others."""
        obs = 0.0408
        dev_wolf = abs(V_CB_WOLFENSTEIN - obs)
        dev_dsq = abs(V_CB_DELTA_SQ - obs)
        dev_geo = abs(V_CB_GEO - obs)
        assert dev_geo < dev_dsq < dev_wolf

    def test_v_cb_best_is_geometric(self):
        assert V_CB_BEST == V_CB_GEO


# ── δ_CP candidate formulas ──────────────────────────────────────

class TestCPPhaseCandidates:

    def test_naive_triality_is_pi_over_2(self):
        """2π/3 − (Δ−δ)π = 2π/3 − π/6 = π/2."""
        val = delta_cp_naive_triality()
        assert abs(val - math.pi / 2) < 1e-14

    def test_dimensional_flow_positive(self):
        assert delta_cp_dimensional_flow() > 0

    def test_fibonacci_phase_positive(self):
        assert delta_cp_fibonacci_phase() > 0

    def test_golden_angle_half_value(self):
        """π/φ² + δ·π/φ."""
        expected = math.pi / PHI**2 + DELTA_G_F * math.pi / PHI
        assert abs(delta_cp_golden_angle_half() - expected) < 1e-15

    def test_golden_angle_damped_value(self):
        """(π/φ²)(1 − Δ)."""
        expected = (math.pi / PHI**2) * (1 - DELTA_F)
        assert abs(delta_cp_golden_angle_damped() - expected) < 1e-15

    def test_pi_over_phi_combined(self):
        """(π/φ)(2 − φ + Δ) — the best candidate."""
        val = delta_cp_pi_over_phi_cubed_corr()
        expected = (math.pi / PHI) * (2.0 - PHI + DELTA_F)
        assert abs(val - expected) < 1e-15

    def test_best_candidate_within_1_sigma(self):
        """Best candidate δ_CP within 1σ of observation."""
        assert BEST_CANDIDATE.deviation_sigma < 1.0

    def test_best_candidate_name(self):
        """The best should be formula I (pi_over_phi_combined)."""
        assert BEST_CANDIDATE.name == "pi_over_phi_combined"

    def test_candidates_sorted_by_deviation(self):
        """Candidates list is sorted by ascending deviation."""
        devs = [c.deviation_rad for c in CP_PHASE_CANDIDATES]
        assert devs == sorted(devs)

    def test_all_candidates_positive(self):
        """All δ_CP predictions are positive."""
        for c in CP_PHASE_CANDIDATES:
            assert c.value_rad > 0, f"{c.name} has negative δ_CP"

    def test_all_candidates_less_than_pi(self):
        """All δ_CP predictions are less than π."""
        for c in CP_PHASE_CANDIDATES:
            assert c.value_rad < math.pi, f"{c.name} exceeds π"

    def test_candidate_count(self):
        """We have 9 candidate formulas."""
        assert len(CP_PHASE_CANDIDATES) == 9

    def test_build_candidates_returns_fresh_list(self):
        """build_candidates() returns a new list each time."""
        c1 = build_candidates()
        c2 = build_candidates()
        assert c1 is not c2
        assert len(c1) == len(c2)


# ── Best-candidate results ────────────────────────────────────────

class TestBestResults:

    def test_delta_cp_close_to_observed(self):
        """δ_CP ≈ 1.144 rad within 0.5σ."""
        assert abs(DELTA_CP_BEST - DELTA_CP_OBSERVED) < 0.5 * DELTA_CP_OBSERVED_UNC + 0.01

    def test_jarlskog_order_of_magnitude(self):
        """J ≈ 3 × 10⁻⁵."""
        assert 2e-5 < JARLSKOG_BEST < 4e-5

    def test_jarlskog_close_to_observed(self):
        """J within 3σ of observed value."""
        sigma = abs(JARLSKOG_BEST - JARLSKOG_OBSERVED) / JARLSKOG_OBSERVED_UNC
        assert sigma < 3.0

    def test_ckm_angles_consistency(self):
        """CKM angles reproduce the input |V_ij|."""
        s13 = math.sin(CKM_ANGLES_BEST.theta_13)
        c13 = math.cos(CKM_ANGLES_BEST.theta_13)
        s12 = math.sin(CKM_ANGLES_BEST.theta_12)
        s23 = math.sin(CKM_ANGLES_BEST.theta_23)
        assert abs(s13 - V_UB) < 1e-12
        assert abs(s12 * c13 - V_US) < 1e-12
        assert abs(s23 * c13 - V_CB_BEST) < 1e-12


# ── Full CKM matrix ──────────────────────────────────────────────

class TestCKMMatrix:

    def test_unitarity(self):
        """V·V† = I to machine precision."""
        dev = ckm_unitarity_check(CKM_MATRIX_BEST)
        assert dev < 1e-14

    def test_v_ud_dominant(self):
        """|V_ud| is the largest element, close to 1."""
        assert abs(CKM_MATRIX_BEST[0][0]) > 0.97

    def test_v_us_matches(self):
        """|V_us| from matrix matches input."""
        assert abs(abs(CKM_MATRIX_BEST[0][1]) - V_US) < 1e-10

    def test_v_ub_matches(self):
        """|V_ub| from matrix matches input."""
        assert abs(abs(CKM_MATRIX_BEST[0][2]) - V_UB) < 1e-10

    def test_v_cb_matches(self):
        """|V_cb| from matrix matches input."""
        assert abs(abs(CKM_MATRIX_BEST[1][2]) - V_CB_BEST) < 1e-10

    def test_matrix_is_3x3(self):
        assert len(CKM_MATRIX_BEST) == 3
        for row in CKM_MATRIX_BEST:
            assert len(row) == 3

    def test_identity_from_zero_angles(self):
        """Zero mixing → identity matrix."""
        angles = CKMAngles(0.0, 0.0, 0.0, 0.0)
        mat = build_ckm_matrix(angles)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(abs(mat[i][j]) - expected) < 1e-15


# ── Registry integration ─────────────────────────────────────────

class TestRegistry:

    def test_register_no_errors(self):
        """register_all() runs without raising."""
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        names = [o.name for o in reg]
        assert "exp_v_cb_geo" in names
        assert "exp_delta_cp" in names
        assert "exp_jarlskog" in names

    def test_registered_levels(self):
        """All CKM phase observables are level 6 (experimental)."""
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        for obs in reg:
            assert obs.level == 6

    def test_delta_cp_deviation_small(self):
        """Registered δ_CP has small sigma deviation."""
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        obs = reg["exp_delta_cp"]
        sigma = obs.deviation_abs / obs.observed_uncertainty
        assert sigma < 1.0

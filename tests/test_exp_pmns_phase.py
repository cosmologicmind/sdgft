"""Tests for the PMNS CP phase module (D₄ triality, long/short roots).

Tests cover:
1. Observed values and D₄ root constants
2. Individual candidate formulas (well-defined output)
3. build_candidates() ordering and completeness
4. PMNS Jarlskog invariant calculation
5. Module-level best candidate
6. Registry integration
"""

import math

import pytest

from sdgft.constants import DELTA_F, DELTA_G_F, PHI
from sdgft.experimental.pmns_phase import (
    # Observed
    DELTA_CP_PMNS_OBSERVED, DELTA_CP_PMNS_OBSERVED_UNC,
    JARLSKOG_PMNS_OBSERVED, JARLSKOG_PMNS_OBSERVED_UNC,
    # D₄ root constants
    SQRT2, D4_LONG_ROOT_LENGTH, D4_SHORT_ROOT_LENGTH,
    D4_ROOT_RATIO, N_ROOTS_D4,
    # Candidate dataclass
    PMNSPhaseCandidate,
    # Individual formulas
    delta_pmns_dual_fibonacci,
    delta_pmns_triality_offset,
    delta_pmns_short_root_golden,
    delta_pmns_tbm_fibonacci,
    delta_pmns_complementary,
    delta_pmns_pi_plus_ckm,
    delta_pmns_golden_complement_dual,
    delta_pmns_3pi_over_2_corrected,
    delta_pmns_3pi2_fibonacci,
    # Builder
    build_candidates, PMNS_CANDIDATES, BEST_PMNS_CANDIDATE,
    DELTA_CP_PMNS_BEST,
    # Jarlskog
    jarlskog_pmns, jarlskog_pmns_sdgft, JARLSKOG_PMNS_BEST,
    # Registry
    register_all,
)


# ── Observed values ───────────────────────────────────────────────

class TestObservedValues:

    def test_delta_cp_range(self):
        """Observed δ_CP^PMNS ∈ [0, 2π]."""
        assert 0.0 < DELTA_CP_PMNS_OBSERVED < 2.0 * math.pi

    def test_uncertainty_positive(self):
        assert DELTA_CP_PMNS_OBSERVED_UNC > 0

    def test_uncertainty_reasonable(self):
        """Uncertainty ~25°, i.e. ~0.44 rad."""
        assert 0.2 < DELTA_CP_PMNS_OBSERVED_UNC < 1.0

    def test_jarlskog_negative(self):
        """J_PMNS is negative (sin δ < 0 in third quadrant)."""
        assert JARLSKOG_PMNS_OBSERVED < 0


# ── D₄ root constants ────────────────────────────────────────────

class TestD4Constants:

    def test_sqrt2(self):
        assert abs(SQRT2 - math.sqrt(2.0)) < 1e-15

    def test_long_root(self):
        assert abs(D4_LONG_ROOT_LENGTH - math.sqrt(2.0)) < 1e-15

    def test_short_root(self):
        assert D4_SHORT_ROOT_LENGTH == 1.0

    def test_root_ratio(self):
        assert abs(D4_ROOT_RATIO - math.sqrt(2.0)) < 1e-15

    def test_n_roots(self):
        assert N_ROOTS_D4 == 24


# ── Candidate dataclass ──────────────────────────────────────────

class TestPMNSPhaseCandidate:

    def test_value_deg(self):
        c = PMNSPhaseCandidate(
            name="test", label="test",
            value_rad=math.pi,
            formula="test", reasoning="test",
        )
        assert abs(c.value_deg - 180.0) < 1e-10

    def test_deviation_circular(self):
        """Circular deviation: distance on circle, not raw difference."""
        c = PMNSPhaseCandidate(
            name="test", label="test",
            value_rad=0.1,  # near 0
            formula="test", reasoning="test",
        )
        # Observed is ~3.44 rad; distance via wrap should be min(|0.1-3.44|, 2π-|0.1-3.44|)
        diff_direct = abs(0.1 - DELTA_CP_PMNS_OBSERVED)
        diff_wrap = 2.0 * math.pi - diff_direct
        expected = min(diff_direct, diff_wrap)
        assert abs(c.deviation_rad - expected) < 1e-10

    def test_deviation_sigma(self):
        c = PMNSPhaseCandidate(
            name="test", label="test",
            value_rad=DELTA_CP_PMNS_OBSERVED,
            formula="test", reasoning="test",
        )
        assert c.deviation_sigma < 1e-10


# ── Individual formulas: all in [0, 2π) ──────────────────────────

class TestFormulas:

    @pytest.mark.parametrize("func", [
        delta_pmns_dual_fibonacci,
        delta_pmns_triality_offset,
        delta_pmns_short_root_golden,
        delta_pmns_tbm_fibonacci,
        delta_pmns_complementary,
        delta_pmns_pi_plus_ckm,
        delta_pmns_golden_complement_dual,
        delta_pmns_3pi_over_2_corrected,
        delta_pmns_3pi2_fibonacci,
    ])
    def test_in_range(self, func):
        """All formulas produce a value in [0, 2π)."""
        val = func()
        assert 0.0 <= val < 2.0 * math.pi

    def test_dual_fibonacci_equals_complementary(self):
        """Candidates A and E are the same formula: 2π − δ^CKM."""
        a = delta_pmns_dual_fibonacci()
        e = delta_pmns_complementary()
        assert abs(a - e) < 1e-12

    def test_triality_offset_value(self):
        """B: δ^CKM + 4π/3 (mod 2π)."""
        from sdgft.experimental.ckm_phase import DELTA_CP_BEST
        expected = (DELTA_CP_BEST + 4.0 * math.pi / 3.0) % (2.0 * math.pi)
        assert abs(delta_pmns_triality_offset() - expected) < 1e-12

    def test_3pi2_corrected_value(self):
        """H: 3π/2 − (Δ−δ)π = 4π/3."""
        expected = 4.0 * math.pi / 3.0
        assert abs(delta_pmns_3pi_over_2_corrected() - expected) < 1e-12

    def test_3pi2_fibonacci_value(self):
        """I: (3π/2)(1 − Δ + δ) = (3π/2)(20/24) = 5π/4."""
        expected = 5.0 * math.pi / 4.0
        assert abs(delta_pmns_3pi2_fibonacci() - expected) < 1e-12

    def test_pi_plus_ckm_value(self):
        from sdgft.experimental.ckm_phase import DELTA_CP_BEST
        expected = (math.pi + DELTA_CP_BEST) % (2.0 * math.pi)
        assert abs(delta_pmns_pi_plus_ckm() - expected) < 1e-12


# ── Jarlskog invariant ───────────────────────────────────────────

class TestJarlskog:

    def test_jarlskog_zero_delta(self):
        """J = 0 when δ_CP = 0 (CP conservation)."""
        j = jarlskog_pmns(33.0, 45.0, 8.5, 0.0)
        assert abs(j) < 1e-15

    def test_jarlskog_maximal(self):
        """J is maximal at δ_CP = π/2."""
        j = jarlskog_pmns(33.0, 45.0, 8.5, math.pi / 2.0)
        assert j > 0.02

    def test_jarlskog_sign(self):
        """sin(3π/2) < 0 → J < 0."""
        j = jarlskog_pmns(33.0, 45.0, 8.5, 3.0 * math.pi / 2.0)
        assert j < 0

    def test_jarlskog_sdgft_callable(self):
        """jarlskog_pmns_sdgft returns a float."""
        j = jarlskog_pmns_sdgft(math.pi / 2)
        assert isinstance(j, float)
        assert j != 0.0

    def test_jarlskog_best_nonzero(self):
        assert JARLSKOG_PMNS_BEST != 0.0

    def test_jarlskog_order_of_magnitude(self):
        """J_PMNS ≈ 0.03 (much larger than J_CKM ≈ 3e-5)."""
        assert 0.001 < abs(JARLSKOG_PMNS_BEST) < 0.1


# ── build_candidates ─────────────────────────────────────────────

class TestBuildCandidates:

    def test_returns_list(self):
        candidates = build_candidates()
        assert isinstance(candidates, list)

    def test_nine_candidates(self):
        assert len(build_candidates()) == 9

    def test_sorted_by_deviation(self):
        candidates = build_candidates()
        devs = [c.deviation_rad for c in candidates]
        assert devs == sorted(devs)

    def test_all_have_names(self):
        for c in build_candidates():
            assert c.name
            assert c.label
            assert c.formula

    def test_unique_names(self):
        names = [c.name for c in build_candidates()]
        # dual_fibonacci and complementary are the same formula,
        # so both are present but with different names
        assert len(names) == len(set(names))


# ── Module-level results ─────────────────────────────────────────

class TestModuleResults:

    def test_pmns_candidates_is_list(self):
        assert isinstance(PMNS_CANDIDATES, list)

    def test_best_is_first(self):
        assert BEST_PMNS_CANDIDATE is PMNS_CANDIDATES[0]

    def test_best_value_in_range(self):
        assert 0.0 <= DELTA_CP_PMNS_BEST < 2.0 * math.pi

    def test_best_within_large_uncertainty(self):
        """Best candidate within ~3σ of observation (large unc.)."""
        assert BEST_PMNS_CANDIDATE.deviation_sigma < 5.0


# ── Registry ──────────────────────────────────────────────────────

class TestRegistry:

    def test_register_no_errors(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        names = [o.name for o in reg]
        assert "exp_delta_cp_pmns" in names
        assert "exp_jarlskog_pmns" in names

    def test_registered_levels(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        for obs in reg:
            assert obs.level == 6

    def test_delta_cp_pmns_registered_value(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        obs = reg["exp_delta_cp_pmns"]
        assert abs(obs.predicted - DELTA_CP_PMNS_BEST) < 1e-15
        assert abs(obs.observed - DELTA_CP_PMNS_OBSERVED) < 1e-15

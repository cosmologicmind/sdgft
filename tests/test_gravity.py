"""Tests for the Horndeski gravity parameters."""

from fractions import Fraction

from sdgft.dimension import N_TREE, N_TREE_F, N_FP
from sdgft.gravity import (
    alpha_m, alpha_b,
    growth_index_analytic, grav_slip,
    ALPHA_M_TREE, ALPHA_M_TREE_F,
    ALPHA_B_TREE, ALPHA_B_TREE_F,
    ALPHA_T, ALPHA_K,
    ALPHA_M_FP, ALPHA_B_FP,
    GAMMA_GROWTH,
    ETA_SLIP_SUBHORIZON, ETA_SLIP_SURVEY, ETA_SLIP_HORIZON,
)


class TestTreeLevel:

    def test_alpha_m_fraction(self):
        assert ALPHA_M_TREE == Fraction(19, 86)

    def test_alpha_m_from_formula(self):
        expected = (N_TREE - 1) / (2 * N_TREE - 1)
        assert ALPHA_M_TREE == expected

    def test_alpha_m_float(self):
        assert abs(ALPHA_M_TREE_F - 19.0 / 86.0) < 1e-15

    def test_alpha_b_fraction(self):
        assert ALPHA_B_TREE == -Fraction(19, 172)

    def test_alpha_b_is_minus_half_alpha_m(self):
        assert ALPHA_B_TREE == -ALPHA_M_TREE / 2

    def test_alpha_t_zero(self):
        assert ALPHA_T == 0

    def test_alpha_k_zero(self):
        assert ALPHA_K == 0


class TestFunctions:

    def test_alpha_m_function_tree(self):
        result = alpha_m(N_TREE_F)
        assert abs(result - ALPHA_M_TREE_F) < 1e-14

    def test_alpha_b_function_tree(self):
        result = alpha_b(N_TREE_F)
        assert abs(result - ALPHA_B_TREE_F) < 1e-14

    def test_alpha_m_function_fp(self):
        result = alpha_m(N_FP)
        assert abs(result - ALPHA_M_FP) < 1e-14


class TestPhysics:

    def test_alpha_m_positive(self):
        """alpha_M > 0 enhances gravity at late times."""
        assert ALPHA_M_TREE_F > 0

    def test_alpha_b_negative(self):
        """alpha_B < 0 provides braiding compensation."""
        assert ALPHA_B_TREE_F < 0

    def test_alpha_m_below_observational_limit(self):
        """Planck MG constraint: alpha_M < 0.36."""
        assert ALPHA_M_TREE_F < 0.36


class TestGrowthIndex:

    def test_gamma_growth_value(self):
        """gamma ~ 0.41 (MCMC numerical result)."""
        assert abs(GAMMA_GROWTH - 0.41) < 1e-10

    def test_gamma_growth_less_than_gr(self):
        """SDGFT predicts less growth than GR (gamma_GR ~ 0.55)."""
        assert GAMMA_GROWTH < 0.55

    def test_gamma_growth_positive(self):
        assert GAMMA_GROWTH > 0

    def test_analytic_formula_gives_different_value(self):
        """The analytic formula (6-3n)/(6n-7) gives ~1.32, not 0.41.
        This documents the monograph vs MCMC discrepancy."""
        analytic = growth_index_analytic(N_TREE_F)
        assert abs(analytic - 29.0 / 22.0) < 1e-10
        assert analytic > 1.0  # clearly wrong for growth index


class TestGravSlip:

    def test_gr_limit(self):
        """n = 1 (GR) gives eta = 1 at all scales."""
        assert abs(grav_slip(1.0, 0.1) - 1.0) < 1e-15
        assert abs(grav_slip(1.0, 10.0) - 1.0) < 1e-15
        assert abs(grav_slip(1.0, 1000.0) - 1.0) < 1e-15

    def test_superhorizon_limit(self):
        """At k/aH -> 0: eta -> 1 (no slip)."""
        assert abs(grav_slip(N_TREE_F, 0.001) - 1.0) < 0.001

    def test_subhorizon_limit(self):
        """At k/aH -> inf: eta -> 1/2."""
        assert abs(grav_slip(N_TREE_F, 10000.0) - 0.5) < 0.001

    def test_subhorizon_fraction(self):
        assert ETA_SLIP_SUBHORIZON == Fraction(1, 2)

    def test_survey_scale(self):
        """At k/aH = 10 (survey scale): eta ~ 0.50."""
        assert abs(ETA_SLIP_SURVEY - 0.503) < 0.01

    def test_horizon_scale(self):
        """At k/aH = 1: eta ~ 0.69."""
        assert abs(ETA_SLIP_HORIZON - 0.694) < 0.01

    def test_eta_less_than_one(self):
        """In SDGFT, eta < 1 on all sub-horizon scales."""
        assert ETA_SLIP_SURVEY < 1.0
        assert ETA_SLIP_HORIZON < 1.0

    def test_eta_greater_than_half(self):
        """At finite k, eta > 1/2."""
        assert ETA_SLIP_SURVEY > 0.5
        assert ETA_SLIP_HORIZON > 0.5

    def test_monotonic_in_scale(self):
        """eta decreases with increasing k/aH."""
        assert grav_slip(N_TREE_F, 1.0) > grav_slip(N_TREE_F, 10.0)
        assert grav_slip(N_TREE_F, 10.0) > grav_slip(N_TREE_F, 100.0)

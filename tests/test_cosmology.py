"""Tests for the cosmological density parameters."""

import math
from fractions import Fraction

from sdgft.cosmology import (
    omega_b_formula, w_de, eta_b, s_8,
    OMEGA_B, OMEGA_B_F,
    OMEGA_C, OMEGA_C_F,
    OMEGA_DE, OMEGA_DE_F,
    OMEGA_M, OMEGA_M_F,
    W_DE_TREE, W_DE_TREE_F, W_DE_FP,
    ETA_B,
    SIGMA_8, SIGMA_8_UNC, S_8,
)
from sdgft.constants import DELTA, DELTA_G, DELTA_G_F


class TestExactFractions:

    def test_omega_b_value(self):
        assert OMEGA_B == Fraction(115, 2304)

    def test_omega_c_value(self):
        assert OMEGA_C == Fraction(600, 2304)

    def test_omega_de_value(self):
        assert OMEGA_DE == Fraction(1589, 2304)

    def test_omega_m_value(self):
        assert OMEGA_M == Fraction(715, 2304)

    def test_flatness_exact(self):
        """Omega_b + Omega_c + Omega_DE = 1 (exact)."""
        assert OMEGA_B + OMEGA_C + OMEGA_DE == 1

    def test_omega_m_is_sum(self):
        assert OMEGA_M == OMEGA_B + OMEGA_C

    def test_w_de_tree_fraction(self):
        assert W_DE_TREE == Fraction(-67, 72)


class TestFormulas:

    def test_omega_b_formula(self):
        result = omega_b_formula(DELTA, DELTA_G)
        assert result == OMEGA_B

    def test_w_de_function(self):
        from sdgft.dimension import D_STAR_TREE_F
        assert abs(w_de(D_STAR_TREE_F) - W_DE_TREE_F) < 1e-14

    def test_eta_b_function(self):
        result = eta_b(DELTA_G_F)
        assert abs(result - ETA_B) < 1e-20


class TestObservationalAgreement:

    def test_omega_b_close_to_observed(self):
        """Omega_b = 0.04991 vs observed 0.0493 +/- 0.0003."""
        assert abs(OMEGA_B_F - 0.0493) < 0.002

    def test_omega_c_close_to_observed(self):
        """Omega_c = 0.2604 vs observed 0.2607 +/- 0.0063."""
        assert abs(OMEGA_C_F - 0.2607) < 0.01

    def test_omega_de_close_to_observed(self):
        """Omega_DE = 0.6897 vs observed 0.6847 +/- 0.0073."""
        assert abs(OMEGA_DE_F - 0.6847) < 0.01

    def test_omega_m_close_to_observed(self):
        """Omega_m = 0.3103 vs observed 0.315 +/- 0.007."""
        assert abs(OMEGA_M_F - 0.315) < 0.01

    def test_w_de_quintessence(self):
        """w_DE > -1 (not cosmological constant)."""
        assert W_DE_TREE_F > -1.0

    def test_w_de_close_to_observed(self):
        """w_DE = -0.931 vs observed -1.03 +/- 0.03."""
        assert abs(W_DE_TREE_F - (-1.03)) < 0.15

    def test_eta_b_order_of_magnitude(self):
        """eta_B should be ~ 6e-10."""
        assert 1e-10 < ETA_B < 1e-9

    def test_eta_b_close_to_observed(self):
        """eta_B = 6.27e-10 (with closure) vs observed 6.104e-10."""
        assert abs(ETA_B - 6.104e-10) < 0.3e-10


class TestSigma8:

    def test_sigma_8_value(self):
        """sigma_8 = 0.775 (MCMC)."""
        assert SIGMA_8 == 0.775

    def test_sigma_8_uncertainty(self):
        assert SIGMA_8_UNC == 0.010

    def test_sigma_8_lower_than_lcdm(self):
        """sigma_8(SDGFT) = 0.775 < sigma_8(LCDM) ~ 0.803."""
        assert SIGMA_8 < 0.803

    def test_sigma_8_agrees_with_lensing(self):
        """Within 1 sigma of DES-Y3 (0.776 +/- 0.017)."""
        assert abs(SIGMA_8 - 0.776) < 0.017

    def test_s_8_value(self):
        """S_8 = sigma_8 * sqrt(Omega_m / 0.3) ~ 0.791."""
        expected = 0.775 * math.sqrt(OMEGA_M_F / 0.3)
        assert abs(S_8 - expected) < 1e-10

    def test_s_8_close_to_0_791(self):
        """Monograph ch05 eq:S8_sdgft quotes 0.791."""
        assert abs(S_8 - 0.791) < 0.005

    def test_s_8_function(self):
        result = s_8(0.775, OMEGA_M_F)
        assert abs(result - S_8) < 1e-14

    def test_s_8_below_planck_cmb(self):
        """S_8(SDGFT) ~ 0.791 < S_8(Planck CMB) ~ 0.832."""
        assert S_8 < 0.832

"""Tests for the dimensional flow module."""

import math
from fractions import Fraction

from sdgft.dimensional_flow import (
    beta_dim, d_star_efold, d_star_of_r, omega_de_rg,
    GAMMA_DE_TREE, GAMMA_DE_TREE_F,
    OMEGA_DE_RG, D_STAR_GALACTIC,
    D_STAR_UV, D_STAR_START, D_STAR_START_F,
    D_STAR_N0, D_STAR_N30, D_STAR_N60,
    R_PLANCK, R_HUBBLE,
)
from sdgft.dimension import D_STAR_TREE_F
from sdgft.constants import DELTA_F, DELTA_G_F


class TestBetaFunction:

    def test_fixed_point_zero(self):
        """Beta = 0 at D* = D*_IR (stable fixed point)."""
        assert abs(beta_dim(67.0 / 24.0, 67.0 / 24.0, 5.0 / 24.0)) < 1e-15

    def test_trivial_fixed_point(self):
        """Beta has another zero at D* = 0 (but not physical)."""
        # At D*=0: 0/0 * (D*_IR - 0) → limit is Delta * D*_IR / D* → ∞
        # We just test that beta > 0 for D* < D*_IR (flow toward IR)
        assert beta_dim(2.0, 67.0 / 24.0, 5.0 / 24.0) > 0

    def test_positive_for_d_star_below_ir(self):
        """dD*/dN > 0 when D* < D*_IR (flowing toward fixed point)."""
        for d in [1.5, 2.0, 2.5]:
            assert beta_dim(d, D_STAR_TREE_F, DELTA_F) > 0

    def test_negative_for_d_star_above_ir(self):
        """dD*/dN < 0 when D* > D*_IR (returning to fixed point)."""
        assert beta_dim(3.0, D_STAR_TREE_F, DELTA_F) < 0


class TestDStarEfold:

    def test_initial_condition(self):
        """D*(N=0) = D*_start."""
        assert abs(D_STAR_N0 - D_STAR_START_F) < 1e-10

    def test_ir_convergence(self):
        """D* approaches D*_IR after many e-folds."""
        d_1000 = d_star_efold(1000.0, D_STAR_TREE_F, D_STAR_START_F, DELTA_F)
        assert abs(d_1000 - D_STAR_TREE_F) < 1e-6

    def test_monotonic_increase(self):
        """D* increases monotonically during inflation."""
        assert D_STAR_N0 < D_STAR_N30 < D_STAR_N60

    def test_n60_close_to_ir(self):
        """After 60 e-folds, D* ~ 2.78 (close to but not exactly D*_IR)."""
        assert abs(D_STAR_N60 - D_STAR_TREE_F) < 0.02
        assert D_STAR_N60 < D_STAR_TREE_F  # not yet converged

    def test_start_value(self):
        """D*_start = 49/24 ~ 2.042."""
        assert D_STAR_START == Fraction(49, 24)
        assert abs(D_STAR_START_F - 2.04167) < 0.001


class TestDarkEnergyRG:

    def test_gamma_de_fraction(self):
        """gamma_DE = delta_g^2 / D* = 1/1608."""
        assert GAMMA_DE_TREE == Fraction(1, 1608)

    def test_gamma_de_value(self):
        assert abs(GAMMA_DE_TREE_F - 0.000622) < 0.00001

    def test_omega_de_at_planck(self):
        """Omega_DE(r_P) = 3/4 (UV boundary)."""
        result = omega_de_rg(R_PLANCK, D_STAR_TREE_F, DELTA_G_F)
        assert abs(result - 0.75) < 1e-10

    def test_omega_de_at_hubble(self):
        """Omega_DE(r_H) ~ 0.687 (monograph: 0.6867)."""
        assert abs(OMEGA_DE_RG - 0.687) < 0.002

    def test_omega_de_decreases_with_scale(self):
        """Omega_DE decreases from UV to IR."""
        ode_small = omega_de_rg(1e-20, D_STAR_TREE_F, DELTA_G_F)
        ode_large = omega_de_rg(1e20, D_STAR_TREE_F, DELTA_G_F)
        assert ode_small > ode_large

    def test_omega_de_close_to_exact_fraction(self):
        """RG value 0.687 is within 0.5% of exact 1589/2304 = 0.690."""
        exact = 1589.0 / 2304.0
        assert abs(OMEGA_DE_RG - exact) / exact < 0.005


class TestPhysicalConstants:

    def test_planck_length(self):
        assert abs(R_PLANCK - 1.616e-35) < 1e-37

    def test_hubble_radius(self):
        assert abs(R_HUBBLE - 8.8e26) < 1e25

    def test_hierarchy(self):
        """r_H / r_P ~ 5.5e61."""
        ratio = R_HUBBLE / R_PLANCK
        assert 5e61 < ratio < 6e61


class TestDStarOfR:
    """Tests for the power-law dimensional flow D*(r) = D*_IR * (r/r_P)^{-Delta^2}."""

    def test_at_planck_scale(self):
        """D*(r_P) = D*_IR (trivial at reference scale)."""
        result = d_star_of_r(R_PLANCK)
        assert abs(result - D_STAR_TREE_F) < 1e-10

    def test_decreases_with_scale(self):
        """D*(r) decreases as r grows (negative exponent)."""
        d_small = d_star_of_r(1e-20)
        d_large = d_star_of_r(1e20)
        assert d_small > d_large

    def test_exponent_is_delta_squared(self):
        """The running exponent is Delta^2 = (5/24)^2 ~ 0.0434."""
        r_test = 1.0  # 1 metre
        result = d_star_of_r(r_test)
        expected = D_STAR_TREE_F * (r_test / R_PLANCK) ** (-(DELTA_F ** 2))
        assert abs(result - expected) < 1e-12

    def test_galactic_value_positive(self):
        """D* at ~1.8 kpc is positive and smaller than D*_IR."""
        assert 0 < D_STAR_GALACTIC < D_STAR_TREE_F

    def test_galactic_value_diagnostic(self):
        """D_STAR_GALACTIC is a module-level constant."""
        r_gal = 1.8e3 * 3.0857e19  # 1.8 kpc in metres
        assert abs(D_STAR_GALACTIC - d_star_of_r(r_gal)) < 1e-12

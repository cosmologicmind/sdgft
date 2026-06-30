"""Tests for the fundamental SDGFT constants."""

import math
from fractions import Fraction

from sdgft.constants import (
    DELTA, DELTA_F, DELTA_G, DELTA_G_F,
    PHI, THETA_MAX, SIN2_30, SIN2_30_F, COS2_30,
)


class TestAxioms:

    def test_delta_value(self):
        assert DELTA == Fraction(5, 24)

    def test_delta_float(self):
        assert abs(DELTA_F - 5.0 / 24.0) < 1e-15

    def test_delta_g_value(self):
        assert DELTA_G == Fraction(1, 24)

    def test_delta_g_float(self):
        assert abs(DELTA_G_F - 1.0 / 24.0) < 1e-15


class TestEmergent:

    def test_phi_golden_ratio(self):
        expected = (1.0 + math.sqrt(5.0)) / 2.0
        assert abs(PHI - expected) < 1e-15

    def test_phi_satisfies_equation(self):
        """phi^2 = phi + 1 (defining property of golden ratio)."""
        assert abs(PHI**2 - PHI - 1.0) < 1e-14

    def test_theta_max(self):
        assert THETA_MAX == 30.0

    def test_sin2_30_fraction(self):
        assert SIN2_30 == Fraction(1, 4)

    def test_sin2_30_agrees_with_trig(self):
        assert abs(SIN2_30_F - math.sin(math.radians(30.0))**2) < 1e-15

    def test_cos2_30(self):
        assert COS2_30 == Fraction(3, 4)

    def test_sin2_cos2_sum(self):
        assert SIN2_30 + COS2_30 == 1


class TestAxiomRelations:

    def test_delta_sum(self):
        """Delta + delta = sin^2(30) = 1/4."""
        assert DELTA + DELTA_G == SIN2_30

    def test_delta_ratio(self):
        """Delta / delta = 5 (Fibonacci index)."""
        assert DELTA / DELTA_G == 5

    def test_golden_ratio_from_ratio(self):
        """phi = (1 + sqrt(Delta/delta)) / 2."""
        ratio = float(DELTA / DELTA_G)
        phi_derived = (1.0 + math.sqrt(ratio)) / 2.0
        assert abs(PHI - phi_derived) < 1e-15

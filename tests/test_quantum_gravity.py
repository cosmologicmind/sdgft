"""Tests for the quantum gravity module."""

import math
import pytest

from sdgft.constants import DELTA_F
from sdgft.physical_constants import G_N, K_P
from sdgft.quantum_gravity import (
    ETA_ANOMALOUS, D_STAR_UV, ETA_LV,
    EPSILON_GAL, G_AT_IR, G_AT_PLANCK,
    g_running, g_radial,
)


class TestAnomalousDimension:
    """Test the graviton anomalous dimension."""

    def test_eta_value(self):
        """eta = 2.0 exactly."""
        assert ETA_ANOMALOUS == 2.0

    def test_eta_vs_cdt(self):
        """CDT: eta = 2.0 +/- 0.3."""
        assert abs(ETA_ANOMALOUS - 2.0) < 0.3


class TestUVFixedPoint:
    """Test the UV spectral dimension."""

    def test_d_star_uv_value(self):
        """D*_UV = 2.0."""
        assert D_STAR_UV == 2.0

    def test_d_star_uv_vs_cdt(self):
        """CDT: D*_UV = 1.80 +/- 0.25. Our 2.0 is within 1 sigma."""
        sigma = abs(D_STAR_UV - 1.80) / 0.25
        assert sigma < 1.0


class TestLorentzViolation:
    """Test the Lorentz violation parameter."""

    def test_eta_lv_formula(self):
        """eta_LV = Delta^2 = (5/24)^2."""
        assert abs(ETA_LV - DELTA_F ** 2) < 1e-15

    def test_eta_lv_numeric(self):
        """eta_LV = 25/576 ~ 0.0434."""
        assert abs(ETA_LV - 25.0 / 576.0) < 1e-15

    def test_eta_lv_below_fermi_lat(self):
        """Fermi-LAT: eta_LV < 0.1."""
        assert ETA_LV < 0.1

    def test_eta_lv_positive(self):
        assert ETA_LV > 0


class TestRunningG:
    """Test the running gravitational coupling."""

    def test_ir_limit(self):
        """At very low k, G -> G_N."""
        g_ir = g_running(1e-30)
        assert abs(g_ir / G_N - 1.0) < 1e-10

    def test_uv_suppressed(self):
        """At k >> k_P, G -> 0."""
        g_uv = g_running(1e10 * K_P)
        assert g_uv < 1e-10 * G_N

    def test_planck_scale(self):
        """At k = k_P, G = G_N / 2."""
        assert abs(G_AT_PLANCK - G_N / 2.0) < 1e-20 * G_N

    def test_monotonic_decrease(self):
        """G(k) decreases with increasing k."""
        scales = [1.0, 1e10, 1e20, 1e30, K_P, 10 * K_P]
        g_vals = [g_running(k) for k in scales]
        for i in range(1, len(g_vals)):
            assert g_vals[i] <= g_vals[i - 1] + 1e-30

    def test_positive(self):
        """G is always positive."""
        for k in [1.0, K_P, 100 * K_P]:
            assert g_running(k) > 0


class TestRadialG:
    """Test the radial parametrization."""

    def test_at_reference_scale(self):
        """G(r_s) = G_N (no correction at reference)."""
        assert abs(g_radial(1.0, r_s=1.0) - G_N) < 1e-25

    def test_enhancement(self):
        """G increases at r > r_s."""
        g_outer = g_radial(10.0, r_s=1.0)
        assert g_outer > G_N

    def test_suppression(self):
        """G decreases at r < r_s."""
        g_inner = g_radial(0.1, r_s=1.0)
        assert g_inner < G_N

    def test_epsilon_effect(self):
        """Larger epsilon gives larger modification."""
        g1 = g_radial(10.0, epsilon=0.1)
        g2 = g_radial(10.0, epsilon=0.2)
        assert g2 > g1

"""Tests for the black holes module."""

import math
import pytest

from sdgft.physical_constants import G_N, C, HBAR, K_B, M_P, R_P, M_SUN
from sdgft.gravity import ALPHA_M_TREE_F
from sdgft.experimental.black_holes import (
    D_STAR_UV, ETA_ANOMALOUS,
    g_running, g_of_r,
    schwarzschild_radius,
    kretschner_classical,
    KRETSCHNER_MAX, KRETSCHNER_MAX_ALT,
    hawking_temperature, T_HAWKING_MAX,
    M_REMNANT, M_REMNANT_GEV,
    qnm_correction,
    bekenstein_hawking_entropy,
    T_HAWKING_SOLAR, QNM_CORRECTION_30MSUN, S_BH_SOLAR,
)


class TestRunningG:
    """Test the running gravitational coupling."""

    def test_g_ir_limit(self):
        """At very low k, G -> G_N."""
        k_ir = 1.0  # 1/m, deep IR
        g = g_running(k_ir)
        # At k << k_P, the correction is negligible
        assert abs(g - G_N) / G_N < 1e-60

    def test_g_planck_scale(self):
        """At k = k_P, G = G_N / 2."""
        k_p = 1.0 / R_P
        g = g_running(k_p)
        assert abs(g - G_N / 2.0) / G_N < 1e-10

    def test_g_uv_suppressed(self):
        """At high k >> k_P, G -> 0."""
        k_uv = 1e3 / R_P
        g = g_running(k_uv)
        assert g < G_N * 1e-4

    def test_g_of_r_large_r(self):
        """At large r, G(r) ~ G_N."""
        g = g_of_r(1.0)  # 1 meter
        assert abs(g - G_N) / G_N < 1e-60

    def test_g_of_r_planck_scale(self):
        """At r = r_P, G = G_N / 2."""
        g = g_of_r(R_P)
        assert abs(g - G_N / 2.0) / G_N < 1e-10

    def test_g_positive(self):
        """G is always positive."""
        for k in [1.0, 1e10, 1e20, 1e30, 1e35]:
            assert g_running(k) > 0

    def test_g_monotonically_decreasing(self):
        """G(k) decreases with k."""
        k_vals = [1.0, 1e10, 1e20, 1e30, 1e35]
        g_vals = [g_running(k) for k in k_vals]
        for i in range(len(g_vals) - 1):
            assert g_vals[i] >= g_vals[i + 1]


class TestSchwarzschildRadius:
    """Test the Schwarzschild radius calculation."""

    def test_solar_mass(self):
        """r_s for 1 M_sun ~ 2953 m."""
        r_s = schwarzschild_radius(M_SUN)
        assert abs(r_s - 2953.0) < 5.0  # within ~5 m

    def test_proportional_to_mass(self):
        """r_s scales linearly with mass."""
        r1 = schwarzschild_radius(M_SUN)
        r2 = schwarzschild_radius(10.0 * M_SUN)
        assert abs(r2 / r1 - 10.0) < 1e-10


class TestKretschner:
    """Test Kretschner scalar and its saturation."""

    def test_kretschner_max_finite(self):
        """Maximum Kretschner scalar is finite (no singularity)."""
        assert math.isfinite(KRETSCHNER_MAX)
        assert KRETSCHNER_MAX > 0

    def test_kretschner_max_order(self):
        """K_max ~ 10^138 m^{-4}."""
        log_k = math.log10(KRETSCHNER_MAX)
        assert 130 < log_k < 145

    def test_kretschner_alt_consistent(self):
        """Two expressions for K_max should agree within an order of magnitude."""
        ratio = KRETSCHNER_MAX / KRETSCHNER_MAX_ALT
        assert 0.1 < ratio < 10.0

    def test_classical_diverges(self):
        """Classical Kretschner grows as r -> 0."""
        k1 = kretschner_classical(M_SUN, 1000.0)
        k2 = kretschner_classical(M_SUN, 100.0)
        assert k2 > k1  # Grows as 1/r^6


class TestHawkingTemperature:
    """Test Hawking temperature with and without running G."""

    def test_classical_solar(self):
        """Classical Hawking T for 1 M_sun ~ 6e-8 K."""
        t = hawking_temperature(M_SUN, use_running_g=False)
        assert abs(t - 6.17e-8) / 6.17e-8 < 0.05  # within 5%

    def test_classical_module_value(self):
        """Module-level value matches function call."""
        t = hawking_temperature(M_SUN, use_running_g=False)
        assert abs(T_HAWKING_SOLAR - t) / t < 1e-10

    def test_t_max_finite(self):
        """T_max is finite."""
        assert math.isfinite(T_HAWKING_MAX)

    def test_t_max_order(self):
        """T_max ~ 10^32 K."""
        log_t = math.log10(T_HAWKING_MAX)
        assert 30 < log_t < 34

    def test_sdgft_bounded(self):
        """SDGFT temperature doesn't diverge for small masses."""
        # A very light black hole should still have finite temperature
        m_small = 10.0 * M_P
        t = hawking_temperature(m_small, use_running_g=True)
        assert math.isfinite(t)
        assert t > 0

    def test_classical_inversely_proportional(self):
        """Classical T_H ~ 1/M."""
        t1 = hawking_temperature(M_SUN, use_running_g=False)
        t2 = hawking_temperature(10.0 * M_SUN, use_running_g=False)
        assert abs(t1 / t2 - 10.0) < 1e-6


class TestPlanckRemnant:
    """Test the Planck-mass remnant prediction."""

    def test_remnant_is_planck_mass(self):
        """Remnant mass = Planck mass."""
        assert M_REMNANT == M_P

    def test_remnant_value(self):
        """M_rem ~ 2.18e-8 kg."""
        assert abs(M_REMNANT - 2.176e-8) / 2.176e-8 < 0.01

    def test_remnant_gev_positive(self):
        """Remnant mass in GeV is positive and large."""
        assert M_REMNANT_GEV > 1e18  # above 10^18 GeV


class TestQNMCorrection:
    """Test quasi-normal mode frequency corrections."""

    def test_qnm_30msun(self):
        """QNM correction for 30 M_sun is negligible."""
        assert QNM_CORRECTION_30MSUN < 1e-70

    def test_qnm_scales_as_r_p_squared(self):
        """Correction scales as (r_P/r_s)^2 ~ 1/M^2."""
        c1 = qnm_correction(M_SUN)
        c2 = qnm_correction(10.0 * M_SUN)
        ratio = c1 / c2
        assert abs(ratio - 100.0) < 1.0  # (10)^2 = 100

    def test_qnm_positive(self):
        """QNM correction is positive (frequency increases)."""
        assert qnm_correction(M_SUN) > 0

    def test_qnm_uses_alpha_m(self):
        """QNM correction proportional to alpha_M."""
        c1 = qnm_correction(M_SUN, alpha_m=ALPHA_M_TREE_F)
        c2 = qnm_correction(M_SUN, alpha_m=2 * ALPHA_M_TREE_F)
        assert abs(c2 / c1 - 2.0) < 1e-10


class TestBekensteinHawking:
    """Test Bekenstein-Hawking entropy."""

    def test_solar_entropy_order(self):
        """S_BH for 1 M_sun ~ 10^{77}."""
        log_s = math.log10(S_BH_SOLAR)
        assert 76 < log_s < 79

    def test_entropy_scales_as_m_squared(self):
        """S ~ M^2 (since A ~ r_s^2 ~ M^2)."""
        s1 = bekenstein_hawking_entropy(M_SUN)
        s2 = bekenstein_hawking_entropy(10.0 * M_SUN)
        ratio = s2 / s1
        assert abs(ratio - 100.0) < 1.0


class TestUVConstants:
    """Test UV fixed-point constants."""

    def test_d_star_uv(self):
        """UV dimension = 2."""
        assert D_STAR_UV == 2.0

    def test_eta_anomalous(self):
        """Anomalous dimension = 2."""
        assert ETA_ANOMALOUS == 2.0


class TestRegistry:
    """Test observable registration."""

    def test_register_all_creates_entries(self):
        """Registration creates observables without error."""
        from sdgft.registry import Registry
        from sdgft.experimental.black_holes import register_all
        reg = Registry()
        register_all(reg)
        assert len(reg) == 3

    def test_observable_names(self):
        """Check expected observable names."""
        from sdgft.registry import Registry
        from sdgft.experimental.black_holes import register_all
        reg = Registry()
        register_all(reg)
        assert "exp_kretschner_max" in reg
        assert "exp_t_hawking_max" in reg
        assert "exp_m_remnant" in reg

"""Tests for the chiral gravity module."""

import math
import pytest

from sdgft.constants import DELTA_F, DELTA_G_F, PHI
from sdgft.experimental.chiral_gravity import (
    chiral_coupling,
    XI_G,
    gw_amplitude_asymmetry,
    gw_asymmetry_smbh,
    GW_ASYMMETRY_LISA,
    GW_ASYMMETRY_LIGO,
)


class TestChiralCoupling:
    """Test the chiral coupling constant xi_G."""

    def test_xi_g_positive(self):
        """xi_G is positive."""
        assert XI_G > 0

    def test_xi_g_small(self):
        """xi_G << 1 (perturbative)."""
        assert XI_G < 0.01

    def test_xi_g_formula(self):
        """xi_G = Delta * delta_g * phi^{-2}."""
        expected = DELTA_F * DELTA_G_F * PHI ** (-2)
        assert abs(XI_G - expected) < 1e-15

    def test_xi_g_value(self):
        """xi_G ~ 0.00332."""
        assert abs(XI_G - 0.00332) < 0.0005

    def test_xi_g_module_level_matches_function(self):
        """Module-level constant matches function call."""
        assert abs(XI_G - chiral_coupling()) < 1e-15

    def test_xi_g_below_upper_bound(self):
        """xi_G < 0.01 (observational upper limit)."""
        assert XI_G < 0.01


class TestParameterDependence:
    """Test sensitivity to input parameters."""

    def test_proportional_to_delta(self):
        """xi_G scales linearly with Delta."""
        xi1 = chiral_coupling(delta=DELTA_F)
        xi2 = chiral_coupling(delta=2 * DELTA_F)
        assert abs(xi2 / xi1 - 2.0) < 1e-10

    def test_proportional_to_delta_g(self):
        """xi_G scales linearly with delta_g."""
        xi1 = chiral_coupling(delta_g=DELTA_G_F)
        xi2 = chiral_coupling(delta_g=2 * DELTA_G_F)
        assert abs(xi2 / xi1 - 2.0) < 1e-10

    def test_inversely_proportional_to_phi_squared(self):
        """xi_G scales as phi^{-2}."""
        xi1 = chiral_coupling(phi=1.0)
        xi2 = chiral_coupling(phi=2.0)
        assert abs(xi2 / xi1 - 0.25) < 1e-10  # (2)^{-2} / (1)^{-2} = 1/4

    def test_zero_delta_gives_zero(self):
        """With Delta=0, xi_G vanishes."""
        xi = chiral_coupling(delta=0.0)
        assert abs(xi) < 1e-20

    def test_zero_delta_g_gives_zero(self):
        """With delta_g=0, xi_G vanishes."""
        xi = chiral_coupling(delta_g=0.0)
        assert abs(xi) < 1e-20


class TestGWAsymmetry:
    """Test gravitational wave parity violation predictions."""

    def test_asymmetry_positive(self):
        """Asymmetry is positive at any frequency."""
        assert gw_amplitude_asymmetry(100.0) > 0

    def test_asymmetry_proportional_to_frequency(self):
        """Asymmetry scales linearly with frequency."""
        a1 = gw_amplitude_asymmetry(100.0)
        a2 = gw_amplitude_asymmetry(200.0)
        assert abs(a2 / a1 - 2.0) < 1e-10

    def test_asymmetry_proportional_to_xi_g(self):
        """Asymmetry scales linearly with xi_G."""
        a1 = gw_amplitude_asymmetry(100.0, xi_g=XI_G)
        a2 = gw_amplitude_asymmetry(100.0, xi_g=2 * XI_G)
        assert abs(a2 / a1 - 2.0) < 1e-10

    def test_ligo_band_unobservable(self):
        """At 100 Hz (LIGO), asymmetry is negligibly small."""
        assert GW_ASYMMETRY_LIGO < 1e-40

    def test_lisa_band_small(self):
        """At mHz (LISA), asymmetry is still very small."""
        assert GW_ASYMMETRY_LISA < 1e-15

    def test_lisa_module_value(self):
        """Module-level LISA value matches function call."""
        val = gw_asymmetry_smbh(1e6)
        assert abs(GW_ASYMMETRY_LISA - val) / val < 1e-10

    def test_smbh_heavier_means_lower_frequency(self):
        """Heavier SMBH -> lower frequency -> smaller asymmetry."""
        a1 = gw_asymmetry_smbh(1e6)
        a2 = gw_asymmetry_smbh(1e7)
        assert a2 < a1  # Higher mass = lower freq = less asymmetry


class TestRegistry:
    """Test observable registration."""

    def test_register_all_creates_entry(self):
        """Registration creates observable without error."""
        from sdgft.registry import Registry
        from sdgft.experimental.chiral_gravity import register_all
        reg = Registry()
        register_all(reg)
        assert len(reg) == 1

    def test_observable_name(self):
        """Check expected observable name."""
        from sdgft.registry import Registry
        from sdgft.experimental.chiral_gravity import register_all
        reg = Registry()
        register_all(reg)
        assert "exp_xi_g" in reg

    def test_is_upper_limit(self):
        """The observed value is an upper limit."""
        from sdgft.registry import Registry
        from sdgft.experimental.chiral_gravity import register_all
        reg = Registry()
        register_all(reg)
        obs = reg.get("exp_xi_g")
        assert obs.is_upper_limit

    def test_predicted_below_limit(self):
        """Predicted xi_G is below the observational upper limit."""
        from sdgft.registry import Registry
        from sdgft.experimental.chiral_gravity import register_all
        reg = Registry()
        register_all(reg)
        obs = reg.get("exp_xi_g")
        assert obs.predicted < obs.observed

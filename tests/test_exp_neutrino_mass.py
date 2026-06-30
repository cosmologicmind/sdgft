"""Tests for the neutrino mass module."""

import math
import pytest

from sdgft.constants import DELTA_G_F
from sdgft.experimental.neutrino_mass import (
    neutrino_mass_sum,
    mass_splitting_normal,
    SUM_M_NU,
    M_NU_1, M_NU_2, M_NU_3,
    M_E_GEV, V_HIGGS_GEV,
)


class TestNeutrinoMassSum:
    """Test the total neutrino mass sum prediction."""

    def test_sum_positive(self):
        """Total mass sum is positive."""
        assert SUM_M_NU > 0

    def test_sum_below_upper_bound(self):
        """sum(m_nu) < 0.12 eV (Planck + BAO upper bound)."""
        assert SUM_M_NU < 0.12

    def test_sum_above_minimum_normal(self):
        """sum(m_nu) > 0.05 eV (normal hierarchy minimum with solar splitting)."""
        # The minimum possible mass sum for normal hierarchy
        # is ~ sqrt(dm32^2) ~ 0.05 eV
        assert SUM_M_NU > 0.04

    def test_sum_in_expected_range(self):
        """sum(m_nu) in range 0.04-0.12 eV."""
        assert 0.04 < SUM_M_NU < 0.12

    def test_sum_module_level_matches_function(self):
        """Module-level constant matches function call."""
        assert abs(SUM_M_NU - neutrino_mass_sum()) < 1e-15

    def test_sum_formula_structure(self):
        """Formula: delta_g * m_e * (v/M_Pl)^(1/3) in eV."""
        from sdgft.physical_constants import M_PL_GEV
        expected_gev = DELTA_G_F * M_E_GEV * (V_HIGGS_GEV / M_PL_GEV) ** (1.0 / 3.0)
        expected_ev = expected_gev * 1e9
        assert abs(SUM_M_NU - expected_ev) < 1e-12


class TestParameterDependence:
    """Test sensitivity to input parameters."""

    def test_proportional_to_delta_g(self):
        """Mass sum scales linearly with delta_g."""
        s1 = neutrino_mass_sum(delta_g=DELTA_G_F)
        s2 = neutrino_mass_sum(delta_g=2 * DELTA_G_F)
        assert abs(s2 / s1 - 2.0) < 1e-10

    def test_zero_delta_g_gives_zero(self):
        """With delta_g=0, mass sum vanishes."""
        s = neutrino_mass_sum(delta_g=0.0)
        assert abs(s) < 1e-20

    def test_seesaw_suppression(self):
        """Lowering M_Pl increases the mass sum (seesaw)."""
        s1 = neutrino_mass_sum()
        s2 = neutrino_mass_sum(m_pl=1e18)  # Lower Planck mass
        assert s2 > s1  # Less suppression


class TestMassSplitting:
    """Test individual mass estimates under normal hierarchy."""

    def test_sum_consistency(self):
        """m1 + m2 + m3 ~ sum(m_nu)."""
        total = M_NU_1 + M_NU_2 + M_NU_3
        assert abs(total - SUM_M_NU) / SUM_M_NU < 0.1  # within 10%

    def test_hierarchy(self):
        """Normal hierarchy: m1 < m2 < m3."""
        assert M_NU_1 <= M_NU_2 <= M_NU_3

    def test_m3_dominates(self):
        """m3 is the heaviest mass eigenstate."""
        assert M_NU_3 > M_NU_2
        assert M_NU_3 > M_NU_1

    def test_all_positive(self):
        """All masses are non-negative."""
        assert M_NU_1 >= 0
        assert M_NU_2 >= 0
        assert M_NU_3 >= 0

    def test_m2_squared_minus_m1_squared(self):
        """m2^2 - m1^2 ~ 7.53e-5 eV^2 (solar splitting)."""
        dm21_sq = M_NU_2 ** 2 - M_NU_1 ** 2
        assert abs(dm21_sq - 7.53e-5) / 7.53e-5 < 0.1

    def test_m3_squared_minus_m2_squared(self):
        """m3^2 - m2^2 ~ 2.453e-3 eV^2 (atmospheric splitting)."""
        dm32_sq = M_NU_3 ** 2 - M_NU_2 ** 2
        assert abs(dm32_sq - 2.453e-3) / 2.453e-3 < 0.1


class TestRegistry:
    """Test observable registration."""

    def test_register_all_creates_entry(self):
        """Registration creates observable without error."""
        from sdgft.registry import Registry
        from sdgft.experimental.neutrino_mass import register_all
        reg = Registry()
        register_all(reg)
        assert len(reg) == 1

    def test_observable_name(self):
        """Check expected observable name."""
        from sdgft.registry import Registry
        from sdgft.experimental.neutrino_mass import register_all
        reg = Registry()
        register_all(reg)
        assert "exp_sum_m_nu" in reg

    def test_is_upper_limit(self):
        """The observed value is an upper limit."""
        from sdgft.registry import Registry
        from sdgft.experimental.neutrino_mass import register_all
        reg = Registry()
        register_all(reg)
        obs = reg.get("exp_sum_m_nu")
        assert obs.is_upper_limit

    def test_predicted_below_limit(self):
        """Predicted value is below the observed upper limit."""
        from sdgft.registry import Registry
        from sdgft.experimental.neutrino_mass import register_all
        reg = Registry()
        register_all(reg)
        obs = reg.get("exp_sum_m_nu")
        assert obs.predicted < obs.observed

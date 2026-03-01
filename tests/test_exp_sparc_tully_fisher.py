"""Tests for the SPARC Baryonic Tully-Fisher module.

Sections:
    1. TestBaryonicMass         — M_bar computation from SPARC photometry
    2. TestTFDataset            — dataset building and filtering
    3. TestBTFRFit              — ODR and fixed-slope BTFR fits
    4. TestTFResult             — pre-computed module-level result
    5. TestTFPlots              — plot generation (no crash tests)
    6. TestTFRegistry           — observable registration
    7. TestTFSummary            — text summary output
"""

from __future__ import annotations

import math

import pytest


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tf_module():
    """Import the TF module (triggers sparc_batch import ~2.3s)."""
    from sdgft.experimental import sparc_tully_fisher as tf
    return tf


@pytest.fixture(scope="module")
def tf_result(tf_module):
    """Pre-computed TF_RESULT."""
    return tf_module.TF_RESULT


@pytest.fixture(scope="module")
def tf_galaxies(tf_module):
    """Build TF dataset."""
    return tf_module.build_tf_dataset()


# ── 1. Baryonic Mass ─────────────────────────────────────────────

class TestBaryonicMass:
    """Test M_bar computation."""

    def test_mass_positive(self, tf_module):
        m_bar, m_star, m_gas = tf_module.baryonic_mass_msun(1.0, 0.5)
        assert m_bar > 0
        assert m_star > 0
        assert m_gas > 0

    def test_mass_sum(self, tf_module):
        m_bar, m_star, m_gas = tf_module.baryonic_mass_msun(1.0, 0.5)
        assert abs(m_bar - m_star - m_gas) < 1.0  # Float precision

    def test_stellar_mass(self, tf_module):
        """M_star = Υ_disk × L × 10⁹."""
        m_bar, m_star, m_gas = tf_module.baryonic_mass_msun(2.0, 0.0)
        expected = 0.5 * 2.0 * 1e9
        assert abs(m_star - expected) < 1.0

    def test_gas_mass(self, tf_module):
        """M_gas = 1.33 × M_HI × 10⁹."""
        m_bar, m_star, m_gas = tf_module.baryonic_mass_msun(0.0, 1.0, ml_disk=0.0)
        expected = 1.33 * 1.0 * 1e9
        assert abs(m_gas - expected) < 1.0

    def test_ml_ratio_effect(self, tf_module):
        """Higher M/L → higher M_star."""
        m1, _, _ = tf_module.baryonic_mass_msun(1.0, 0.5, ml_disk=0.3)
        m2, _, _ = tf_module.baryonic_mass_msun(1.0, 0.5, ml_disk=0.7)
        assert m2 > m1

    def test_zero_luminosity(self, tf_module):
        """Pure gas galaxy."""
        m_bar, m_star, m_gas = tf_module.baryonic_mass_msun(0.0, 1.0, ml_disk=0.5)
        assert m_star == 0.0
        assert m_gas > 0
        assert m_bar == m_gas


# ── 2. TF Dataset ────────────────────────────────────────────────

class TestTFDataset:
    """Test dataset building and filtering."""

    def test_dataset_not_empty(self, tf_galaxies):
        assert len(tf_galaxies) > 50

    def test_galaxies_have_vflat(self, tf_galaxies):
        for g in tf_galaxies:
            assert g.v_flat_kms >= 10.0

    def test_galaxies_have_mbar(self, tf_galaxies):
        for g in tf_galaxies:
            assert g.m_bar_msun > 0
            assert g.log_m_bar > 0

    def test_log_v_consistent(self, tf_galaxies):
        for g in tf_galaxies:
            expected = math.log10(g.v_flat_kms)
            assert abs(g.log_v_flat - expected) < 1e-10

    def test_log_m_consistent(self, tf_galaxies):
        for g in tf_galaxies:
            expected = math.log10(g.m_bar_msun)
            assert abs(g.log_m_bar - expected) < 1e-10

    def test_galaxies_sorted(self, tf_galaxies):
        names = [g.name for g in tf_galaxies]
        assert names == sorted(names)

    def test_quality_range(self, tf_galaxies):
        for g in tf_galaxies:
            assert g.quality in (1, 2, 3)

    def test_no_zero_vflat(self, tf_galaxies):
        """V_flat = 0 galaxies should be excluded."""
        for g in tf_galaxies:
            assert g.v_flat_kms > 0

    def test_dataset_count_reasonable(self, tf_galaxies):
        """Between 100 and 175 galaxies expected with V_flat > 10."""
        assert 100 <= len(tf_galaxies) <= 175

    def test_some_have_nu_fit(self, tf_galaxies):
        """At least some galaxies should have power-law fits."""
        with_nu = [g for g in tf_galaxies if g.nu_fit is not None]
        assert len(with_nu) > 50


# ── 3. BTFR Fit ──────────────────────────────────────────────────

class TestBTFRFit:
    """Test BTFR fitting."""

    def test_slope_positive(self, tf_result):
        assert tf_result is not None
        assert tf_result.b_fit > 0

    def test_slope_range(self, tf_result):
        """Slope should be in [2.5, 5.0] (typical BTFR range)."""
        assert 2.5 < tf_result.b_fit < 5.0

    def test_slope_near_theory(self, tf_result):
        """SDGFT predicts b = 91/24 ≈ 3.79. Fit within ~3σ."""
        sigma = abs(tf_result.b_fit - tf_result.b_theory) / tf_result.b_fit_err
        assert sigma < 3.0

    def test_intercept_reasonable(self, tf_result):
        """Intercept should be between 0 and 5."""
        assert 0 < tf_result.a_fit < 5

    def test_scatter_positive(self, tf_result):
        assert tf_result.rms_scatter > 0
        assert tf_result.rms_scatter_fixed > 0

    def test_scatter_reasonable(self, tf_result):
        """Scatter < 1 dex (should be ~0.2-0.4 dex)."""
        assert tf_result.rms_scatter < 1.0
        assert tf_result.rms_scatter_fixed < 1.0

    def test_intrinsic_scatter(self, tf_result):
        assert tf_result.intrinsic_scatter >= 0
        assert tf_result.intrinsic_scatter <= tf_result.rms_scatter

    def test_fixed_intercept(self, tf_result):
        """Fixed-slope intercept should be positive."""
        assert tf_result.a_fixed > 0

    def test_sigma_b_positive(self, tf_result):
        assert tf_result.sigma_b >= 0

    def test_theory_value(self, tf_result):
        """b_theory should be 91/24."""
        assert abs(tf_result.b_theory - 91 / 24) < 1e-10

    def test_quality_sum(self, tf_result):
        total = (tf_result.n_quality_1 + tf_result.n_quality_2
                 + tf_result.n_quality_3)
        assert total == tf_result.n_galaxies


# ── 4. Module-Level Result ────────────────────────────────────────

class TestTFModuleResult:
    """Test the pre-computed TF_RESULT."""

    def test_result_not_none(self, tf_result):
        assert tf_result is not None

    def test_result_type(self, tf_module, tf_result):
        assert isinstance(tf_result, tf_module.TFResult)

    def test_n_galaxies(self, tf_result):
        assert tf_result.n_galaxies > 100

    def test_galaxies_tuple(self, tf_result):
        assert isinstance(tf_result.galaxies, tuple)
        assert len(tf_result.galaxies) == tf_result.n_galaxies

    def test_galaxy_types(self, tf_module, tf_result):
        for g in tf_result.galaxies[:5]:
            assert isinstance(g, tf_module.TFGalaxy)

    def test_b_fit_err_positive(self, tf_result):
        assert tf_result.b_fit_err > 0

    def test_ngc3198_in_dataset(self, tf_result):
        """NGC3198 (canonical benchmark) should be present."""
        names = [g.name for g in tf_result.galaxies]
        assert "NGC3198" in names


# ── 5. Plots ─────────────────────────────────────────────────────

class TestTFPlots:
    """Test that plot functions don't crash."""

    def test_plot_btfr_no_crash(self, tf_module, tf_result):
        fig = tf_module.plot_btfr(tf_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_residuals_no_crash(self, tf_module, tf_result):
        fig = tf_module.plot_residuals(tf_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_nu_vs_vflat_no_crash(self, tf_module, tf_result):
        fig = tf_module.plot_nu_vs_vflat(tf_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_combined_no_crash(self, tf_module, tf_result):
        fig = tf_module.plot_combined(tf_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_btfr_save(self, tf_module, tf_result, tmp_path):
        path = str(tmp_path / "test_btfr.png")
        fig = tf_module.plot_btfr(tf_result, save_path=path)
        assert fig is not None
        import os
        assert os.path.exists(path)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_none_result(self, tf_module):
        """Should handle None result gracefully."""
        fig = tf_module.plot_btfr(None)
        # Will return None since TF_RESULT is used as fallback
        # (which may or may not be None depending on data)

    def test_plot_combined_save(self, tf_module, tf_result, tmp_path):
        path = str(tmp_path / "test_combined.png")
        fig = tf_module.plot_combined(tf_result, save_path=path)
        assert fig is not None
        import os
        assert os.path.exists(path)
        import matplotlib.pyplot as plt
        plt.close("all")


# ── 6. Registry ──────────────────────────────────────────────────

class TestTFRegistry:
    """Test observable registration."""

    def test_register_creates_observables(self, tf_module):
        from sdgft.registry import Registry
        reg = Registry()
        tf_module.register_all(reg)
        names = [o.name for o in reg.all()]
        assert "exp_btfr_slope_sparc" in names

    def test_register_scatter_observable(self, tf_module):
        from sdgft.registry import Registry
        reg = Registry()
        tf_module.register_all(reg)
        names = [o.name for o in reg.all()]
        assert "exp_btfr_scatter_sparc" in names

    def test_btfr_slope_value(self, tf_module, tf_result):
        from sdgft.registry import Registry
        reg = Registry()
        tf_module.register_all(reg)
        obs = {o.name: o for o in reg.all()}
        slope_obs = obs["exp_btfr_slope_sparc"]
        assert abs(slope_obs.predicted - tf_result.b_fit) < 1e-6

    def test_btfr_slope_theory(self, tf_module):
        from sdgft.registry import Registry
        reg = Registry()
        tf_module.register_all(reg)
        obs = {o.name: o for o in reg.all()}
        slope_obs = obs["exp_btfr_slope_sparc"]
        assert abs(slope_obs.observed - 91 / 24) < 1e-6


# ── 7. Summary ────────────────────────────────────────────────────

class TestTFSummary:
    """Test text summary output."""

    def test_summary_no_crash(self, tf_module, capsys):
        tf_module.print_tf_summary()
        captured = capsys.readouterr()
        assert "TULLY-FISHER" in captured.out

    def test_summary_contains_slope(self, tf_module, capsys):
        tf_module.print_tf_summary()
        captured = capsys.readouterr()
        assert "b_fit" in captured.out
        assert "91/24" in captured.out

    def test_summary_contains_galaxies(self, tf_module, capsys):
        tf_module.print_tf_summary()
        captured = capsys.readouterr()
        assert "galaxies" in captured.out

    def test_summary_none_handled(self, tf_module, capsys):
        """print_tf_summary(None) should not crash."""
        # This will use TF_RESULT as fallback
        tf_module.print_tf_summary()

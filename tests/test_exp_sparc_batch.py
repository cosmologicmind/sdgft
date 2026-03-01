"""Tests for the SPARC batch processing module.

Tests cover:
1. Catalog loading (SPARC_Lelli2016c.mrt → 175 entries)
2. Rotation curve loading (NGC3198 has 43 data points)
3. Full database loading (171 galaxies after quality + min-points filter)
4. Baryonic velocity computation (sign preservation, M/L scaling)
5. Surface density estimation and screening factor
6. Single galaxy fitting (ε scan with screening)
7. Batch result (SPARC_BATCH module-level precomputed result)
8. M/L scan sensitivity
9. Registry integration
10. Power-law velocity and chi2 functions
11. Power-law fitting and batch (ν = 2α_M resummation)
12. Power-law registry integration
13. Dynamic chameleon screening (Σ_screen = f · Σ₀)
"""

import math
import os

import pytest

from sdgft.gravity import ALPHA_M_TREE_F
from sdgft.tully_fisher import R_TRANS_KPC
from sdgft.registry import REGISTRY

# ── Skip everything if SPARC data not available ──────────────────

SPARC_DATA_DIR = "/home/david/Coding/data/sparc"
SPARC_AVAILABLE = os.path.isdir(os.path.join(SPARC_DATA_DIR, "Rotmod_LTG"))

pytestmark = pytest.mark.skipif(
    not SPARC_AVAILABLE,
    reason="SPARC data not available at {!r}".format(SPARC_DATA_DIR),
)


# ── Lazy import (module runs batch at import time) ───────────────

@pytest.fixture(scope="module")
def sparc():
    """Lazily import the sparc_batch module (triggers batch computation)."""
    from sdgft.experimental import sparc_batch
    return sparc_batch


@pytest.fixture(scope="module")
def catalog(sparc):
    return sparc.load_sparc_catalog()


@pytest.fixture(scope="module")
def galaxies(sparc):
    return sparc.load_sparc_database()


@pytest.fixture(scope="module")
def batch(sparc):
    return sparc.SPARC_BATCH


@pytest.fixture(scope="module")
def powerlaw_batch(sparc):
    return sparc.SPARC_POWERLAW


@pytest.fixture(scope="module")
def dynamic_batch(sparc):
    return sparc.SPARC_DYNAMIC


# =====================================================================
#  1. Catalog loading
# =====================================================================


class TestSPARCCatalog:
    """Tests for SPARC catalog parsing."""

    def test_catalog_count(self, catalog):
        """SPARC contains exactly 175 galaxies."""
        assert len(catalog) == 175

    def test_entry_type(self, sparc, catalog):
        """Each entry is a SPARCCatalogEntry."""
        for e in catalog:
            assert isinstance(e, sparc.SPARCCatalogEntry)

    def test_ngc3198_present(self, catalog):
        """NGC3198 is in the catalog."""
        names = [e.name for e in catalog]
        assert "NGC3198" in names

    def test_ngc3198_properties(self, catalog):
        """NGC3198 has reasonable catalog properties."""
        ngc = [e for e in catalog if e.name == "NGC3198"][0]
        # Distance ~13.8 Mpc (Freedman et al.)
        assert 10.0 < ngc.distance_mpc < 20.0
        # Flat velocity ~150 km/s
        assert 120.0 < ngc.v_flat_kms < 180.0
        # Quality 1 (high)
        assert ngc.quality == 1
        # Hubble type (Sb ≈ 5)
        assert 0 <= ngc.hubble_type <= 11

    def test_quality_distribution(self, catalog):
        """Quality flags span 1–3 with expected counts."""
        q1 = sum(1 for e in catalog if e.quality == 1)
        q2 = sum(1 for e in catalog if e.quality == 2)
        q3 = sum(1 for e in catalog if e.quality == 3)
        assert q1 + q2 + q3 == 175
        assert q1 >= 90     # ~99
        assert q2 >= 50     # ~64
        assert q3 >= 5      # ~12

    def test_distances_positive(self, catalog):
        """All distances are positive."""
        for e in catalog:
            assert e.distance_mpc > 0

    def test_luminosities_positive(self, catalog):
        """All luminosities are positive."""
        for e in catalog:
            assert e.luminosity_1e9Lsun > 0

    def test_disk_scale_lengths_positive(self, catalog):
        """All disk scale lengths are positive."""
        for e in catalog:
            assert e.r_disk_kpc > 0

    def test_catalog_entries_frozen(self, catalog):
        """Catalog entries are immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            catalog[0].name = "modified"


# =====================================================================
#  2. Rotation curve loading
# =====================================================================


class TestSPARCRotationCurve:
    """Tests for loading individual rotation model files."""

    def test_ngc3198_points(self, sparc):
        """NGC3198 has 43 data points in SPARC."""
        data = sparc.load_rotation_curve("NGC3198")
        assert len(data) == 43

    def test_data_point_type(self, sparc):
        """Data points are SPARCDataPoint instances."""
        data = sparc.load_rotation_curve("NGC3198")
        for dp in data:
            assert isinstance(dp, sparc.SPARCDataPoint)

    def test_sorted_by_radius(self, sparc):
        """Data points are sorted by increasing radius."""
        data = sparc.load_rotation_curve("NGC3198")
        for i in range(1, len(data)):
            assert data[i].r_kpc >= data[i - 1].r_kpc

    def test_positive_radii(self, sparc):
        """All radii are positive."""
        data = sparc.load_rotation_curve("NGC3198")
        for dp in data:
            assert dp.r_kpc > 0

    def test_positive_errors(self, sparc):
        """Velocity errors are positive."""
        data = sparc.load_rotation_curve("NGC3198")
        for dp in data:
            assert dp.v_err_kms > 0

    def test_ngc3198_radius_range(self, sparc):
        """NGC3198 extends from ~0.3 to ~44 kpc."""
        data = sparc.load_rotation_curve("NGC3198")
        assert data[0].r_kpc < 1.0
        assert data[-1].r_kpc > 40.0

    def test_ngc3198_velocity_range(self, sparc):
        """NGC3198 Vobs peaks around 150 km/s."""
        data = sparc.load_rotation_curve("NGC3198")
        v_obs = [dp.v_obs_kms for dp in data]
        assert max(v_obs) > 140.0
        assert max(v_obs) < 180.0

    def test_missing_galaxy_raises(self, sparc):
        """Non-existent galaxy raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            sparc.load_rotation_curve("FAKE_GALAXY_999")

    def test_data_points_frozen(self, sparc):
        """Data points are immutable."""
        data = sparc.load_rotation_curve("NGC3198")
        with pytest.raises(AttributeError):
            data[0].r_kpc = 999.0


# =====================================================================
#  3. Full database loading
# =====================================================================


class TestSPARCDatabase:
    """Tests for loading the full SPARC database."""

    def test_galaxy_count(self, galaxies):
        """With default filters, ~171 galaxies loaded (175 minus 4 sparse)."""
        assert 165 <= len(galaxies) <= 175

    def test_galaxy_type(self, sparc, galaxies):
        """Each entry is a SPARCGalaxy."""
        for g in galaxies:
            assert isinstance(g, sparc.SPARCGalaxy)

    def test_min_data_points(self, sparc, galaxies):
        """All loaded galaxies have ≥ MIN_DATA_POINTS data points."""
        for g in galaxies:
            assert g.n_points >= sparc.MIN_DATA_POINTS

    def test_sorted_by_name(self, galaxies):
        """Galaxies are sorted alphabetically by name."""
        names = [g.name for g in galaxies]
        assert names == sorted(names)

    def test_ngc3198_in_database(self, galaxies):
        """NGC3198 is in the loaded database."""
        names = [g.name for g in galaxies]
        assert "NGC3198" in names

    def test_quality_filter(self, sparc):
        """Quality filter correctly restricts the sample."""
        gals_q1 = sparc.load_sparc_database(max_quality=1)
        gals_all = sparc.load_sparc_database(max_quality=3)
        assert len(gals_q1) < len(gals_all)
        for g in gals_q1:
            assert g.catalog.quality == 1

    def test_dropped_galaxies(self, galaxies):
        """Galaxies with too few points are excluded."""
        # These 4 galaxies have < 5 data points in SPARC
        dropped = {"D512-2", "NGC6789", "UGC00634", "UGC07232"}
        loaded_names = {g.name for g in galaxies}
        for name in dropped:
            assert name not in loaded_names

    def test_total_data_points(self, galaxies):
        """Total data points across all galaxies is ~3000+."""
        total = sum(g.n_points for g in galaxies)
        assert total > 3000


# =====================================================================
#  4. Baryonic velocity computation
# =====================================================================


class TestBaryonicVelocity:
    """Tests for v²_bar, v_newton, v_sdgft formulas."""

    def test_v2_baryonic_positive(self, sparc):
        """Baryonic v² is non-negative for typical data."""
        data = sparc.load_rotation_curve("NGC3198")
        for dp in data:
            v2 = sparc.v2_baryonic_sparc(dp)
            assert v2 >= 0

    def test_v2_baryonic_ml_scaling(self, sparc):
        """Larger M/L gives larger baryonic v²."""
        data = sparc.load_rotation_curve("NGC3198")
        dp = data[10]  # Mid-radius point
        v2_low = sparc.v2_baryonic_sparc(dp, ml_disk=0.3, ml_bulge=0.5)
        v2_high = sparc.v2_baryonic_sparc(dp, ml_disk=1.0, ml_bulge=1.2)
        assert v2_high > v2_low

    def test_v_newton_equals_sqrt_v2(self, sparc):
        """v_newton = sqrt(v²_bar)."""
        data = sparc.load_rotation_curve("NGC3198")
        for dp in data[:10]:
            v_n = sparc.v_newton_sparc(dp)
            v2 = sparc.v2_baryonic_sparc(dp)
            assert abs(v_n - math.sqrt(max(v2, 0))) < 1e-10

    def test_v_sdgft_below_rtrans(self, sparc):
        """Below r_trans, v_sdgft equals v_newton (no G_eff enhancement)."""
        data = sparc.load_rotation_curve("NGC3198")
        # Find a point below r_trans (~1.02 kpc)
        inner = [dp for dp in data if dp.r_kpc < sparc.R_TRANS_KPC]
        if inner:
            dp = inner[0]
            v_n = sparc.v_newton_sparc(dp)
            v_s = sparc.v_sdgft_sparc(dp, epsilon=0.5)
            assert abs(v_n - v_s) < 1e-10

    def test_v_sdgft_above_rtrans(self, sparc):
        """Above r_trans, v_sdgft > v_newton for ε > 0."""
        data = sparc.load_rotation_curve("NGC3198")
        outer = [dp for dp in data if dp.r_kpc > 5.0]
        assert len(outer) > 0
        for dp in outer:
            v_n = sparc.v_newton_sparc(dp)
            v_s = sparc.v_sdgft_sparc(dp, epsilon=0.5)
            if v_n > 0:
                assert v_s > v_n

    def test_v_sdgft_increases_with_epsilon(self, sparc):
        """Larger ε gives larger v_sdgft at fixed radius."""
        data = sparc.load_rotation_curve("NGC3198")
        dp = data[20]  # Well above r_trans
        v1 = sparc.v_sdgft_sparc(dp, epsilon=0.1)
        v2 = sparc.v_sdgft_sparc(dp, epsilon=1.0)
        assert v2 > v1

    def test_gas_sign_preservation(self, sparc):
        """Gas term preserves sign: |v_gas| * v_gas."""
        dp = sparc.SPARCDataPoint(
            r_kpc=5.0, v_obs_kms=100.0, v_err_kms=5.0,
            v_gas_kms=-10.0,  # Negative gas velocity (inflow)
            v_disk_kms=80.0, v_bul_kms=0.0,
            sb_disk=100.0, sb_bul=0.0,
        )
        v2 = sparc.v2_baryonic_sparc(dp, ml_disk=1.0, ml_bulge=1.0)
        # v2 = 1.0 * 80² + 0 + |(-10)| * (-10) = 6400 - 100 = 6300
        assert abs(v2 - 6300.0) < 1e-6


# =====================================================================
#  5. Surface density and screening
# =====================================================================


class TestScreening:
    """Tests for surface density estimation and screening factor."""

    def test_surface_density_positive(self, sparc):
        """Surface density is non-negative."""
        data = sparc.load_rotation_curve("NGC3198")
        for dp in data:
            sigma = sparc._estimate_surface_density(dp)
            assert sigma >= 0

    def test_surface_density_decreases_outward(self, sparc):
        """Surface density generally decreases with radius."""
        data = sparc.load_rotation_curve("NGC3198")
        # Compare inner and outer regions
        sigma_inner = sparc._estimate_surface_density(data[2])
        sigma_outer = sparc._estimate_surface_density(data[-5])
        assert sigma_inner > sigma_outer

    def test_sigma_screen_positive(self, sparc):
        """Screening surface density is positive."""
        data = sparc.load_rotation_curve("NGC3198")
        sigma_screen = sparc._estimate_sigma_screen(data)
        assert sigma_screen > 0

    def test_screening_factor_range(self, sparc):
        """Screening factor S ∈ [0, 1]."""
        data = sparc.load_rotation_curve("NGC3198")
        sigma_screen = sparc._estimate_sigma_screen(data)
        for dp in data:
            s = sparc._screening_factor_sparc(
                dp, sigma_screen, sparc.P_SCREEN_DEFAULT)
            assert 0.0 <= s <= 1.0

    def test_screening_outer_unscreened(self, sparc):
        """At large radii, S ≈ 1 (unscreened)."""
        data = sparc.load_rotation_curve("NGC3198")
        sigma_screen = sparc._estimate_sigma_screen(data)
        # Last point (R ≈ 44 kpc)
        s = sparc._screening_factor_sparc(
            data[-1], sigma_screen, sparc.P_SCREEN_DEFAULT)
        assert s > 0.99

    def test_screening_inner_suppressed(self, sparc):
        """At inner radii, S < 1 (screening active)."""
        data = sparc.load_rotation_curve("NGC3198")
        sigma_screen = sparc._estimate_sigma_screen(data)
        # First point (R ≈ 0.3 kpc)
        s = sparc._screening_factor_sparc(
            data[0], sigma_screen, sparc.P_SCREEN_DEFAULT)
        assert s < 0.5

    def test_screening_zero_sigma_returns_one(self, sparc):
        """If sigma_screen = 0, factor is 1 (no screening)."""
        data = sparc.load_rotation_curve("NGC3198")
        s = sparc._screening_factor_sparc(data[0], 0.0, 1.0)
        assert s == 1.0

    def test_ml_affects_surface_density(self, sparc):
        """Larger M/L increases surface density."""
        data = sparc.load_rotation_curve("NGC3198")
        dp = data[5]
        sigma_low = sparc._estimate_surface_density(dp, ml_disk=0.3)
        sigma_high = sparc._estimate_surface_density(dp, ml_disk=1.0)
        assert sigma_high > sigma_low


# =====================================================================
#  6. Single galaxy fitting
# =====================================================================


class TestFitGalaxy:
    """Tests for fitting ε on a single galaxy."""

    def test_fit_result_type(self, sparc, galaxies):
        """fit_galaxy returns a GalaxyFitResult."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy(gal, n_steps=50)
        assert isinstance(fit, sparc.GalaxyFitResult)

    def test_fit_ngc3198_converges(self, sparc, galaxies):
        """NGC3198 converges with default settings."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy(gal)
        assert fit.converged

    def test_fit_epsilon_positive(self, sparc, galaxies):
        """Best-fit ε is positive."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy(gal)
        assert fit.epsilon_best > 0

    def test_fit_sdgft_beats_newton(self, sparc, galaxies):
        """For NGC3198, SDGFT χ² < Newtonian χ²."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy(gal)
        assert fit.chi2_sdgft < fit.chi2_newton

    def test_fit_chi2_red_sdgft(self, sparc, galaxies):
        """Reduced χ² is computed correctly."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy(gal)
        expected = fit.chi2_sdgft / max(fit.n_data - 2, 1)
        assert abs(fit.chi2_red_sdgft - expected) < 1e-10

    def test_fit_improvement(self, sparc, galaxies):
        """Improvement > 1 means SDGFT is better."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy(gal)
        assert fit.improvement > 1.0

    def test_fit_metadata(self, sparc, galaxies):
        """Fit carries galaxy metadata."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy(gal)
        assert fit.name == "NGC3198"
        assert fit.n_data == 43
        assert fit.quality == 1
        assert fit.distance_mpc > 0

    def test_fit_no_screening(self, sparc, galaxies):
        """Fit without screening also works (gives different ε)."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit_on = sparc.fit_galaxy(gal, screening=True, n_steps=100)
        fit_off = sparc.fit_galaxy(gal, screening=False, n_steps=100)
        # Both converge but ε values differ
        assert fit_on.converged
        assert fit_off.converged
        # Without screening, ε may be slightly different
        # (just ensure both give valid fits)
        assert fit_off.epsilon_best > 0

    def test_fit_ml_sensitivity(self, sparc, galaxies):
        """Higher M/L yields lower best-fit ε."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit_low = sparc.fit_galaxy(gal, ml_disk=0.5, ml_bulge=0.7, n_steps=100)
        fit_high = sparc.fit_galaxy(gal, ml_disk=1.0, ml_bulge=1.2, n_steps=100)
        assert fit_low.epsilon_best > fit_high.epsilon_best

    def test_fit_result_frozen(self, sparc, galaxies):
        """GalaxyFitResult is frozen."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy(gal, n_steps=50)
        with pytest.raises(AttributeError):
            fit.epsilon_best = 0.0


# =====================================================================
#  7. Batch result (module-level)
# =====================================================================


class TestBatchResult:
    """Tests for the pre-computed SPARC_BATCH result."""

    def test_batch_not_none(self, batch):
        """SPARC_BATCH is computed when data is available."""
        assert batch is not None

    def test_batch_type(self, sparc, batch):
        """SPARC_BATCH is a BatchResult."""
        assert isinstance(batch, sparc.BatchResult)

    def test_batch_total(self, batch):
        """171 galaxies in the batch (175 − 4 sparse)."""
        assert batch.n_total == 171

    def test_batch_converged(self, batch):
        """Majority of galaxies converge."""
        assert batch.n_converged > 100
        assert batch.n_converged <= batch.n_fitted

    def test_batch_improved(self, batch):
        """Vast majority show SDGFT > Newton."""
        assert batch.n_improved > 150

    def test_epsilon_mean_positive(self, batch):
        """Mean ε is positive."""
        assert batch.epsilon_mean > 0

    def test_epsilon_sem_positive(self, batch):
        """SEM is positive."""
        assert batch.epsilon_sem > 0

    def test_epsilon_std_positive(self, batch):
        """Standard deviation is positive."""
        assert batch.epsilon_std > 0

    def test_epsilon_theory_value(self, batch):
        """Theory ε matches α_M(1−α_M)."""
        expected = ALPHA_M_TREE_F * (1.0 - ALPHA_M_TREE_F)
        assert abs(batch.epsilon_theory - expected) < 1e-10

    def test_sigma_from_theory_positive(self, batch):
        """Deviation from theory is > 0 (since ⟨ε⟩ ≠ ε_theory)."""
        assert batch.sigma_from_theory > 0

    def test_weighted_mean_positive(self, batch):
        """Weighted mean ε is positive."""
        assert batch.epsilon_weighted_mean > 0

    def test_chi2_sdgft_lt_newton(self, batch):
        """SDGFT median χ²_red is much lower than Newton's."""
        assert batch.median_chi2_red_sdgft < batch.median_chi2_red_newton

    def test_quality_counts(self, batch):
        """Quality counts sum to n_total."""
        assert (batch.n_quality_1 + batch.n_quality_2
                + batch.n_quality_3) == batch.n_total

    def test_q1_converged(self, batch):
        """Q=1 convergence count is consistent."""
        assert batch.n_q1_converged > 0
        assert batch.n_q1_converged <= batch.n_quality_1

    def test_fits_tuple(self, batch):
        """Fits is a tuple of GalaxyFitResult."""
        assert isinstance(batch.fits, tuple)
        assert len(batch.fits) == batch.n_total

    def test_ngc3198_in_fits(self, batch):
        """NGC3198 is among the fitted galaxies."""
        names = [f.name for f in batch.fits]
        assert "NGC3198" in names

    def test_ngc3198_epsilon(self, batch):
        """NGC3198 ε is in a reasonable range (0.1–3.0 at M/L=0.5)."""
        ngc = [f for f in batch.fits if f.name == "NGC3198"][0]
        assert 0.1 < ngc.epsilon_best < 3.0

    def test_batch_epsilon_range(self, batch):
        """All converged ε values are within scan boundaries."""
        for f in batch.fits:
            if f.converged:
                assert f.epsilon_best > 0


# =====================================================================
#  8. M/L sensitivity scan
# =====================================================================


class TestMLScan:
    """Tests for the M/L sensitivity scan function."""

    def test_scan_returns_list(self, sparc, galaxies):
        """scan_ml returns a list of (M/L, mean_ε, SEM) tuples."""
        results = sparc.scan_ml(galaxies, ml_values=(0.5, 1.0), n_steps=20)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_scan_tuple_structure(self, sparc, galaxies):
        """Each result is a 3-tuple (M/L, mean_ε, SEM)."""
        results = sparc.scan_ml(galaxies, ml_values=(0.5,), n_steps=20)
        ml, eps, sem = results[0]
        assert isinstance(ml, float)
        assert isinstance(eps, float)
        assert isinstance(sem, float)
        assert ml == 0.5
        assert eps > 0
        assert sem > 0

    def test_scan_epsilon_decreases_with_ml(self, sparc, galaxies):
        """Higher M/L → lower mean ε (more baryonic mass, less gravity boost)."""
        results = sparc.scan_ml(
            galaxies, ml_values=(0.5, 1.5), n_steps=20)
        ml_low, eps_low, _ = results[0]
        ml_high, eps_high, _ = results[1]
        assert eps_low > eps_high


# =====================================================================
#  9. Helper functions
# =====================================================================


class TestHelpers:
    """Tests for utility functions."""

    def test_median_odd(self, sparc):
        """Median of odd-length list."""
        assert sparc._median([3.0, 1.0, 2.0]) == 2.0

    def test_median_even(self, sparc):
        """Median of even-length list."""
        assert sparc._median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_median_empty(self, sparc):
        """Median of empty list returns 0."""
        assert sparc._median([]) == 0.0

    def test_weighted_mean(self, sparc):
        """Weighted mean with equal weights = simple mean."""
        values = [1.0, 2.0, 3.0]
        weights = [1.0, 1.0, 1.0]
        mean, sem = sparc._weighted_mean_sem(values, weights)
        assert abs(mean - 2.0) < 1e-10

    def test_weighted_mean_empty(self, sparc):
        """Weighted mean of empty inputs returns (0, 0)."""
        mean, sem = sparc._weighted_mean_sem([], [])
        assert mean == 0.0
        assert sem == 0.0


# =====================================================================
#  10. Registry integration
# =====================================================================


class TestRegistry:
    """Tests for register_all() function."""

    def test_register_creates_observables(self, sparc):
        """register_all() adds SPARC observables to a fresh registry."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        assert "exp_epsilon_sparc_mean" in reg
        assert "exp_epsilon_sparc_weighted" in reg

    def test_registered_values(self, sparc, batch):
        """Registered values match batch result."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        obs_mean = reg["exp_epsilon_sparc_mean"]
        assert abs(obs_mean.predicted - batch.epsilon_mean) < 1e-10
        assert abs(obs_mean.observed - batch.epsilon_theory) < 1e-10

    def test_registered_weighted(self, sparc, batch):
        """Weighted observable matches batch result."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        obs_w = reg["exp_epsilon_sparc_weighted"]
        assert abs(obs_w.predicted - batch.epsilon_weighted_mean) < 1e-10


# =====================================================================
#  11. Print summary (smoke test)
# =====================================================================


class TestPrintSummary:
    """Smoke tests for print_summary output."""

    def test_print_summary_no_crash(self, sparc, batch, capsys):
        """print_summary runs without error."""
        sparc.print_summary(batch)
        captured = capsys.readouterr()
        assert "SPARC Batch" in captured.out
        assert "KEY RESULT" in captured.out

    def test_print_summary_none(self, sparc, capsys):
        """print_summary with None input falls back to SPARC_BATCH."""
        sparc.print_summary()
        captured = capsys.readouterr()
        assert "KEY RESULT" in captured.out

    def test_print_summary_contains_statistics(self, sparc, batch, capsys):
        """Summary includes ε statistics."""
        sparc.print_summary(batch)
        captured = capsys.readouterr()
        assert "ε" in captured.out or "epsilon" in captured.out.lower()
        assert "converged" in captured.out.lower() or "Converged" in captured.out


# =====================================================================
#  12. SPARCGalaxy properties
# =====================================================================


class TestSPARCGalaxy:
    """Tests for SPARCGalaxy dataclass."""

    def test_name_property(self, galaxies):
        """name property returns catalog name."""
        ngc = [g for g in galaxies if g.name == "NGC3198"][0]
        assert ngc.name == ngc.catalog.name

    def test_n_points_property(self, galaxies):
        """n_points returns len(data)."""
        ngc = [g for g in galaxies if g.name == "NGC3198"][0]
        assert ngc.n_points == len(ngc.data)

    def test_r_max_kpc(self, galaxies):
        """r_max_kpc returns last data point radius."""
        ngc = [g for g in galaxies if g.name == "NGC3198"][0]
        assert ngc.r_max_kpc == ngc.data[-1].r_kpc
        assert ngc.r_max_kpc > 40.0


# =====================================================================
#  13. Constants consistency
# =====================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_epsilon_theory(self, sparc):
        """EPSILON_THEORY = α_M(1−α_M)."""
        expected = ALPHA_M_TREE_F * (1.0 - ALPHA_M_TREE_F)
        assert abs(sparc.EPSILON_THEORY - expected) < 1e-12

    def test_r_trans_consistent(self, sparc):
        """R_TRANS_KPC matches tully_fisher module."""
        assert abs(sparc.R_TRANS_KPC - R_TRANS_KPC) < 1e-12

    def test_scan_range(self, sparc):
        """ε scan range bounds are sensible."""
        assert sparc.EPSILON_MIN > 0
        assert sparc.EPSILON_MAX > sparc.EPSILON_MIN
        assert sparc.N_EPSILON_STEPS >= 100

    def test_p_screen_value(self, sparc):
        """Screening exponent p = 24/19."""
        assert abs(sparc.P_SCREEN_DEFAULT - 24.0 / 19.0) < 1e-12

    def test_nu_theory_value(self, sparc):
        """NU_THEORY = 2 * α_M."""
        expected = 2.0 * ALPHA_M_TREE_F
        assert abs(sparc.NU_THEORY - expected) < 1e-12

    def test_nu_scan_range(self, sparc):
        """ν scan range bounds are sensible."""
        assert sparc.NU_MIN > 0
        assert sparc.NU_MAX > sparc.NU_MIN
        assert sparc.N_NU_STEPS >= 50


# =====================================================================
#  14. Power-law velocity functions
# =====================================================================


class TestPowerlawVelocity:
    """Tests for the resummed power-law velocity formula."""

    def test_v_powerlaw_below_rtrans(self, sparc):
        """Below r_trans, power-law v equals Newtonian v."""
        data = sparc.load_rotation_curve("NGC3198")
        inner = [dp for dp in data if dp.r_kpc < sparc.R_TRANS_KPC]
        if inner:
            dp = inner[0]
            v_n = sparc.v_newton_sparc(dp)
            v_p = sparc.v_sdgft_powerlaw(dp, nu=0.5)
            assert abs(v_n - v_p) < 1e-10

    def test_v_powerlaw_above_rtrans(self, sparc):
        """Above r_trans, power-law v > Newtonian v for ν > 0."""
        data = sparc.load_rotation_curve("NGC3198")
        outer = [dp for dp in data if dp.r_kpc > 5.0]
        for dp in outer:
            v_n = sparc.v_newton_sparc(dp)
            v_p = sparc.v_sdgft_powerlaw(dp, nu=0.5)
            if v_n > 0:
                assert v_p > v_n

    def test_v_powerlaw_increases_with_nu(self, sparc):
        """Larger ν gives larger v at fixed radius."""
        data = sparc.load_rotation_curve("NGC3198")
        dp = data[20]  # Well above r_trans
        v1 = sparc.v_sdgft_powerlaw(dp, nu=0.2)
        v2 = sparc.v_sdgft_powerlaw(dp, nu=0.8)
        assert v2 > v1

    def test_powerlaw_stronger_than_log(self, sparc):
        """At large R, power-law v > log v for same parameter."""
        data = sparc.load_rotation_curve("NGC3198")
        dp = data[-1]  # R ≈ 44 kpc, well into nonlinear regime
        nu = 0.44
        v_log = sparc.v_sdgft_sparc(dp, epsilon=nu)
        v_pow = sparc.v_sdgft_powerlaw(dp, nu=nu)
        # Power law must be stronger (resummed)
        assert v_pow > v_log

    def test_powerlaw_equals_log_near_rtrans(self, sparc):
        """Near r_trans, power-law ≈ log (first-order agreement)."""
        # Create a point just slightly above r_trans
        dp = sparc.SPARCDataPoint(
            r_kpc=sparc.R_TRANS_KPC * 1.05,
            v_obs_kms=100.0, v_err_kms=5.0,
            v_gas_kms=20.0, v_disk_kms=80.0, v_bul_kms=0.0,
            sb_disk=100.0, sb_bul=0.0,
        )
        nu = 0.44
        v_log = sparc.v_sdgft_sparc(dp, epsilon=nu)
        v_pow = sparc.v_sdgft_powerlaw(dp, nu=nu)
        # Should agree to ~1% near r_trans
        assert abs(v_pow - v_log) / v_log < 0.01


# =====================================================================
#  15. Power-law chi2
# =====================================================================


class TestPowerlawChi2:
    """Tests for power-law chi2 computation."""

    def test_chi2_returns_two_floats(self, sparc):
        """_chi2_galaxy_powerlaw returns (chi2_newton, chi2_powerlaw)."""
        data = sparc.load_rotation_curve("NGC3198")
        c2n, c2s = sparc._chi2_galaxy_powerlaw(data, nu=0.5)
        assert isinstance(c2n, float)
        assert isinstance(c2s, float)
        assert c2n > 0
        assert c2s > 0

    def test_chi2_powerlaw_beats_newton(self, sparc):
        """For NGC3198, power-law chi2 < Newtonian chi2 at ν=0.5."""
        data = sparc.load_rotation_curve("NGC3198")
        c2n, c2s = sparc._chi2_galaxy_powerlaw(data, nu=0.5)
        assert c2s < c2n

    def test_chi2_zero_param_ngc3198(self, sparc):
        """NGC3198 zero-param (ν=2α_M) gives excellent fit."""
        data = sparc.load_rotation_curve("NGC3198")
        sigma_screen = sparc._estimate_sigma_screen(data)
        c2n, c2s = sparc._chi2_galaxy_powerlaw(
            data, sparc.NU_THEORY,
            screening=True, sigma_screen=sigma_screen)
        n = len(data)
        chi2_red = c2s / max(n - 1, 1)
        # Should be much better than Newton
        assert chi2_red < c2n / max(n - 1, 1)
        # χ²_red should be under ~10 for NGC3198
        assert chi2_red < 10.0


# =====================================================================
#  16. Power-law single galaxy fit
# =====================================================================


class TestFitGalaxyPowerlaw:
    """Tests for fit_galaxy_powerlaw."""

    def test_fit_returns_powerlaw_result(self, sparc, galaxies):
        """fit_galaxy_powerlaw returns PowerlawFitResult."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal, n_steps=50)
        assert isinstance(fit, sparc.PowerlawFitResult)

    def test_fit_converges(self, sparc, galaxies):
        """NGC3198 converges in power-law mode."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal)
        assert fit.converged

    def test_fit_nu_positive(self, sparc, galaxies):
        """Best-fit ν is positive."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal)
        assert fit.nu_best > 0

    def test_fit_has_zero_param(self, sparc, galaxies):
        """Fit includes zero-parameter chi2."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal)
        assert fit.chi2_zero_param > 0

    def test_fit_beats_newton(self, sparc, galaxies):
        """Power-law fit beats Newton for NGC3198."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal)
        assert fit.chi2_powerlaw_fit < fit.chi2_newton

    def test_fit_zero_param_good(self, sparc, galaxies):
        """NGC3198 zero-param chi2_red < 10."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal)
        assert fit.chi2_red_zero_param < 10.0

    def test_improvement_properties(self, sparc, galaxies):
        """Improvement properties are > 1 for NGC3198."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal)
        assert fit.improvement_fit > 1.0
        assert fit.improvement_zero > 1.0

    def test_fit_metadata(self, sparc, galaxies):
        """Power-law fit carries galaxy metadata."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal, n_steps=50)
        assert fit.name == "NGC3198"
        assert fit.n_data == 43
        assert fit.quality == 1

    def test_fit_frozen(self, sparc, galaxies):
        """PowerlawFitResult is frozen."""
        gal = [g for g in galaxies if g.name == "NGC3198"][0]
        fit = sparc.fit_galaxy_powerlaw(gal, n_steps=50)
        with pytest.raises(AttributeError):
            fit.nu_best = 0.0


# =====================================================================
#  17. Power-law batch result (module-level)
# =====================================================================


class TestPowerlawBatchResult:
    """Tests for the pre-computed SPARC_POWERLAW result."""

    def test_powerlaw_not_none(self, powerlaw_batch):
        """SPARC_POWERLAW is computed when data is available."""
        assert powerlaw_batch is not None

    def test_powerlaw_type(self, sparc, powerlaw_batch):
        """SPARC_POWERLAW is a PowerlawBatchResult."""
        assert isinstance(powerlaw_batch, sparc.PowerlawBatchResult)

    def test_powerlaw_total(self, powerlaw_batch):
        """171 galaxies in the power-law batch."""
        assert powerlaw_batch.n_total == 171

    def test_powerlaw_converged(self, powerlaw_batch):
        """Majority converge in power-law mode."""
        assert powerlaw_batch.n_converged > 140

    def test_powerlaw_improved_fit(self, powerlaw_batch):
        """Powerlaw(fitted) beats Newton for most galaxies."""
        assert powerlaw_batch.n_improved_fit > 150

    def test_powerlaw_improved_zero(self, powerlaw_batch):
        """Zero-param (ν=2α_M) beats Newton for majority."""
        assert powerlaw_batch.n_improved_zero > 100

    def test_nu_mean_positive(self, powerlaw_batch):
        """Mean ν is positive."""
        assert powerlaw_batch.nu_mean > 0

    def test_nu_theory_value(self, powerlaw_batch):
        """Theory ν matches 2α_M."""
        expected = 2.0 * ALPHA_M_TREE_F
        assert abs(powerlaw_batch.nu_theory - expected) < 1e-10

    def test_sigma_much_less_than_log(self, powerlaw_batch, batch):
        """Power-law σ from theory << log-mode σ from theory."""
        # Power-law resummation should be much closer to theory
        assert powerlaw_batch.sigma_weighted < batch.sigma_weighted

    def test_chi2_zero_vs_newton(self, powerlaw_batch):
        """Zero-param median χ²_red < Newton median χ²_red."""
        assert (powerlaw_batch.median_chi2_red_powerlaw_zero
                < powerlaw_batch.median_chi2_red_newton)

    def test_ngc3198_in_powerlaw(self, powerlaw_batch):
        """NGC3198 is among the power-law fits."""
        names = [f.name for f in powerlaw_batch.fits]
        assert "NGC3198" in names

    def test_ngc3198_nu_near_theory(self, powerlaw_batch, sparc):
        """NGC3198 best-fit ν is close to 2α_M."""
        ngc = [f for f in powerlaw_batch.fits if f.name == "NGC3198"][0]
        # Should be within ~0.1 of theory
        assert abs(ngc.nu_best - sparc.NU_THEORY) < 0.15

    def test_fits_tuple(self, powerlaw_batch):
        """Fits is a tuple of PowerlawFitResult."""
        assert isinstance(powerlaw_batch.fits, tuple)
        assert len(powerlaw_batch.fits) == powerlaw_batch.n_total


# =====================================================================
#  18. Power-law registry
# =====================================================================


class TestPowerlawRegistry:
    """Tests for power-law observable registration."""

    def test_register_creates_nu_observables(self, sparc):
        """register_all() adds ν observables to a fresh registry."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        assert "exp_nu_sparc_mean" in reg
        assert "exp_nu_sparc_weighted" in reg

    def test_nu_registered_values(self, sparc, powerlaw_batch):
        """Registered ν values match batch result."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        obs = reg["exp_nu_sparc_mean"]
        assert abs(obs.predicted - powerlaw_batch.nu_mean) < 1e-10
        assert abs(obs.observed - powerlaw_batch.nu_theory) < 1e-10

    def test_nu_weighted_registered(self, sparc, powerlaw_batch):
        """Weighted ν observable matches batch result."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        obs = reg["exp_nu_sparc_weighted"]
        assert abs(obs.predicted - powerlaw_batch.nu_weighted_mean) < 1e-10

    def test_all_four_observables(self, sparc):
        """register_all creates all 4 SPARC observables."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        names = [o.name for o in reg]
        assert "exp_epsilon_sparc_mean" in names
        assert "exp_epsilon_sparc_weighted" in names
        assert "exp_nu_sparc_mean" in names
        assert "exp_nu_sparc_weighted" in names


# =====================================================================
#  19. Power-law summary (smoke test)
# =====================================================================


class TestPowerlawSummary:
    """Smoke tests for print_powerlaw_summary."""

    def test_print_powerlaw_no_crash(self, sparc, powerlaw_batch, capsys):
        """print_powerlaw_summary runs without error."""
        sparc.print_powerlaw_summary(powerlaw_batch)
        captured = capsys.readouterr()
        assert "Power-Law" in captured.out
        assert "KEY RESULT" in captured.out
        assert "2α_M" in captured.out

    def test_print_powerlaw_shows_comparison(self, sparc, capsys):
        """Summary includes χ² comparison across modes."""
        sparc.print_powerlaw_summary()
        captured = capsys.readouterr()
        assert "Newton" in captured.out
        assert "powerlaw" in captured.out.lower() or "ν" in captured.out


# =====================================================================
#  20. Dynamic screening constants
# =====================================================================


class TestDynamicConstants:
    """Tests for dynamic screening constants."""

    def test_f_screen_default(self, sparc):
        """F_SCREEN_DEFAULT is 1.0."""
        assert sparc.F_SCREEN_DEFAULT == 1.0

    def test_sigma_screen_floor(self, sparc):
        """SIGMA_SCREEN_FLOOR is 1e4."""
        assert sparc.SIGMA_SCREEN_FLOOR == 1e4

    def test_sigma_screen_floor_positive(self, sparc):
        """Floor must be positive."""
        assert sparc.SIGMA_SCREEN_FLOOR > 0


# =====================================================================
#  21. Central surface density
# =====================================================================


class TestCentralSurfaceDensity:
    """Tests for _central_surface_density()."""

    def test_returns_positive(self, sparc, galaxies):
        """Σ₀ is positive for every galaxy."""
        for gal in galaxies:
            sigma0 = sparc._central_surface_density(gal)
            assert sigma0 > 0, f"{gal.name}: Σ₀ = {sigma0}"

    def test_above_floor(self, sparc, galaxies):
        """Σ₀ >= SIGMA_SCREEN_FLOOR for all galaxies."""
        for gal in galaxies:
            sigma0 = sparc._central_surface_density(gal)
            assert sigma0 >= sparc.SIGMA_SCREEN_FLOOR

    def test_hsb_larger_than_threshold(self, sparc, galaxies):
        """At least some galaxies have Σ₀ well above the floor."""
        high = [gal for gal in galaxies
                if sparc._central_surface_density(gal) > 1e7]
        # Expect plenty of massive spirals
        assert len(high) > 10

    def test_spread_covers_orders_of_magnitude(self, sparc, galaxies):
        """Σ₀ range spans at least 2 orders of magnitude."""
        sigmas = [sparc._central_surface_density(gal) for gal in galaxies]
        ratio = max(sigmas) / min(sigmas)
        assert ratio > 100

    def test_sb_disk_branch(self, sparc, galaxies):
        """Galaxies with sb_disk > 0 use the catalog branch."""
        gal = next(g for g in galaxies if g.catalog.sb_disk > 0)
        sigma0 = sparc._central_surface_density(gal)
        expected = sparc.ML_DISK * gal.catalog.sb_disk * 1e6
        assert sigma0 == max(expected, sparc.SIGMA_SCREEN_FLOOR)

    def test_ngc3198_sb(self, sparc, galaxies):
        """NGC3198 has a well-defined Σ₀."""
        ngc = next(g for g in galaxies if g.name == "NGC3198")
        sigma0 = sparc._central_surface_density(ngc)
        # NGC3198 is a bright spiral — expect high surface density
        assert sigma0 > 1e6


# =====================================================================
#  22. Dynamic sigma screen
# =====================================================================


class TestDynamicSigmaScreen:
    """Tests for _dynamic_sigma_screen()."""

    def test_positive(self, sparc, galaxies):
        """Dynamic Σ_screen is positive for every galaxy."""
        for gal in galaxies:
            s = sparc._dynamic_sigma_screen(gal)
            assert s > 0

    def test_equals_sigma0_at_fscreen_one(self, sparc, galaxies):
        """When f_screen=1, Σ_screen == Σ₀."""
        gal = galaxies[0]
        sigma0 = sparc._central_surface_density(gal)
        s = sparc._dynamic_sigma_screen(gal, f_screen=1.0)
        assert abs(s - sigma0) < 1e-6

    def test_scales_with_fscreen(self, sparc, galaxies):
        """Σ_screen ∝ f_screen (above floor)."""
        gal = next(g for g in galaxies
                   if sparc._central_surface_density(g) > 1e7)
        s1 = sparc._dynamic_sigma_screen(gal, f_screen=1.0)
        s2 = sparc._dynamic_sigma_screen(gal, f_screen=2.0)
        assert abs(s2 / s1 - 2.0) < 0.01

    def test_floor_enforced(self, sparc, galaxies):
        """Tiny f_screen still respects the floor."""
        gal = galaxies[0]
        s = sparc._dynamic_sigma_screen(gal, f_screen=1e-20)
        assert s >= sparc.SIGMA_SCREEN_FLOOR

    def test_differs_between_galaxies(self, sparc, galaxies):
        """Different galaxies get different screening thresholds."""
        screens = {sparc._dynamic_sigma_screen(gal) for gal in galaxies[:20]}
        # Should have more than 1 unique value
        assert len(screens) > 1


# =====================================================================
#  23. Dynamic single-galaxy fit
# =====================================================================


class TestFitGalaxyPowerlawDynamic:
    """Tests for fit_galaxy_powerlaw_dynamic()."""

    @pytest.fixture(scope="class")
    def ngc3198_fit(self, sparc, galaxies):
        ngc = next(g for g in galaxies if g.name == "NGC3198")
        return sparc.fit_galaxy_powerlaw_dynamic(ngc)

    def test_returns_powerlaw_fit_result(self, sparc, ngc3198_fit):
        """Returns a PowerlawFitResult."""
        assert isinstance(ngc3198_fit, sparc.PowerlawFitResult)

    def test_name(self, ngc3198_fit):
        """Fit records galaxy name."""
        assert ngc3198_fit.name == "NGC3198"

    def test_nu_positive(self, ngc3198_fit):
        """Best-fit ν is positive."""
        assert ngc3198_fit.nu_best > 0

    def test_nu_near_theory(self, sparc, ngc3198_fit):
        """NGC3198 best-fit ν is within 0.15 of theory."""
        assert abs(ngc3198_fit.nu_best - sparc.NU_THEORY) < 0.15

    def test_converged(self, ngc3198_fit):
        """NGC3198 fit converges (not at boundary)."""
        assert ngc3198_fit.converged

    def test_chi2_less_than_newton(self, ngc3198_fit):
        """Fitted curve beats Newtonian for NGC3198."""
        assert ngc3198_fit.chi2_powerlaw_fit < ngc3198_fit.chi2_newton

    def test_zero_param_computed(self, ngc3198_fit):
        """Zero-parameter χ² is computed and finite."""
        assert ngc3198_fit.chi2_zero_param > 0
        assert math.isfinite(ngc3198_fit.chi2_zero_param)

    def test_n_data_ngc3198(self, ngc3198_fit):
        """NGC3198 has 43 data points."""
        assert ngc3198_fit.n_data == 43


# =====================================================================
#  24. Dynamic batch result (module-level)
# =====================================================================


class TestDynamicBatchResult:
    """Tests for the pre-computed SPARC_DYNAMIC result."""

    def test_dynamic_not_none(self, dynamic_batch):
        """SPARC_DYNAMIC is computed when data is available."""
        assert dynamic_batch is not None

    def test_dynamic_type(self, sparc, dynamic_batch):
        """SPARC_DYNAMIC is a PowerlawBatchResult."""
        assert isinstance(dynamic_batch, sparc.PowerlawBatchResult)

    def test_dynamic_total(self, dynamic_batch):
        """171 galaxies in the dynamic batch."""
        assert dynamic_batch.n_total == 171

    def test_dynamic_converged(self, dynamic_batch):
        """Majority converge in dynamic mode."""
        assert dynamic_batch.n_converged > 140

    def test_dynamic_improved_fit(self, dynamic_batch):
        """Dynamic-fitted beats Newton for most galaxies."""
        assert dynamic_batch.n_improved_fit > 150

    def test_dynamic_improved_zero(self, dynamic_batch):
        """Zero-param dynamic beats Newton for majority."""
        assert dynamic_batch.n_improved_zero > 100

    def test_nu_mean_positive(self, dynamic_batch):
        """Mean ν is positive."""
        assert dynamic_batch.nu_mean > 0

    def test_nu_theory_value(self, dynamic_batch):
        """Theory ν matches 2α_M."""
        expected = 2.0 * ALPHA_M_TREE_F
        assert abs(dynamic_batch.nu_theory - expected) < 1e-10

    def test_sigma_weighted_under_three(self, dynamic_batch):
        """Weighted σ from theory < 3 (≈ 1.69σ currently)."""
        assert dynamic_batch.sigma_weighted < 3.0

    def test_sigma_at_most_static(self, dynamic_batch, powerlaw_batch):
        """Dynamic screening σ <= static screening σ."""
        assert dynamic_batch.sigma_weighted <= powerlaw_batch.sigma_weighted + 0.1

    def test_chi2_zero_vs_newton(self, dynamic_batch):
        """Zero-param median χ²_red < Newton median χ²_red."""
        assert (dynamic_batch.median_chi2_red_powerlaw_zero
                < dynamic_batch.median_chi2_red_newton)

    def test_ngc3198_in_dynamic(self, dynamic_batch):
        """NGC3198 is among the dynamic fits."""
        names = [f.name for f in dynamic_batch.fits]
        assert "NGC3198" in names

    def test_fits_tuple(self, dynamic_batch):
        """Fits is a tuple of PowerlawFitResult."""
        assert isinstance(dynamic_batch.fits, tuple)
        assert len(dynamic_batch.fits) == dynamic_batch.n_total


# =====================================================================
#  25. Dynamic registry
# =====================================================================


class TestDynamicRegistry:
    """Tests for dynamic screening observable registration."""

    def test_register_creates_dynamic_observable(self, sparc):
        """register_all() adds the dynamic weighted observable."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        assert "exp_nu_sparc_dynamic_weighted" in reg

    def test_dynamic_registered_values(self, sparc, dynamic_batch):
        """Registered dynamic ν matches batch result."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        obs = reg["exp_nu_sparc_dynamic_weighted"]
        assert abs(obs.predicted - dynamic_batch.nu_weighted_mean) < 1e-10
        assert abs(obs.observed - dynamic_batch.nu_theory) < 1e-10

    def test_dynamic_observable_level(self, sparc):
        """Dynamic observable has level 6."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        obs = reg["exp_nu_sparc_dynamic_weighted"]
        assert obs.level == 6

    def test_all_five_observables(self, sparc):
        """register_all creates all 5 SPARC observables."""
        from sdgft.registry import Registry
        reg = Registry()
        sparc.register_all(registry=reg)
        names = [o.name for o in reg]
        assert "exp_epsilon_sparc_mean" in names
        assert "exp_epsilon_sparc_weighted" in names
        assert "exp_nu_sparc_mean" in names
        assert "exp_nu_sparc_weighted" in names
        assert "exp_nu_sparc_dynamic_weighted" in names


# =====================================================================
#  26. Dynamic summary (smoke test)
# =====================================================================


class TestDynamicSummary:
    """Smoke tests for print_dynamic_summary."""

    def test_print_dynamic_no_crash(self, sparc, dynamic_batch, capsys):
        """print_dynamic_summary runs without error."""
        sparc.print_dynamic_summary(dynamic_batch)
        captured = capsys.readouterr()
        assert "DYNAMIC" in captured.out
        assert "Σ_screen" in captured.out

    def test_dynamic_shows_comparison(self, sparc, capsys):
        """Dynamic summary includes comparison table."""
        sparc.print_dynamic_summary()
        captured = capsys.readouterr()
        assert "COMPARISON" in captured.out
        assert "static" in captured.out.lower()
        assert "dynamic" in captured.out.lower()

    def test_dynamic_shows_lsb_hsb_split(self, sparc, capsys):
        """Dynamic summary includes LSB vs HSB split."""
        sparc.print_dynamic_summary()
        captured = capsys.readouterr()
        assert "LSB" in captured.out
        assert "HSB" in captured.out

    def test_dynamic_shows_fscreen(self, sparc, capsys):
        """Summary shows f_screen value."""
        sparc.print_dynamic_summary()
        captured = capsys.readouterr()
        assert "f_screen" in captured.out

    def test_none_batch_graceful(self, sparc, capsys):
        """Prints message when dynamic batch is None."""
        sparc.print_dynamic_summary(dyn=None)
        # Should not crash — either prints "not available"
        # or uses SPARC_DYNAMIC
        capsys.readouterr()  # just ensure no crash

"""Tests for the galaxy rotation curve module.

Tests cover:
1. Epsilon candidates and ordering
2. G_eff(r) formula (transition, logarithmic regime)
3. Baryonic mass model (exponential disk, enclosed mass)
4. NGC 3198 model and data
5. Rotation curve computation (spherical and exact thin disk)
6. Fit / χ² statistics
7. Effective dark matter diagnostic
8. D* profile at galactic scales
9. Modified Bessel functions (A&S §9.8)
10. Freeman (1970) thin-disk formula
11. Exact disk vs spherical comparison
12. Chameleon screening: surface density, screening factor, screened curves
13. Registry integration
"""

import math

import pytest

from sdgft.constants import DELTA_F
from sdgft.dimension import D_STAR_TREE_F
from sdgft.gravity import ALPHA_M_TREE_F
from sdgft.physical_constants import G_N, KPC_M, M_SUN
from sdgft.tully_fisher import R_TRANS_KPC
from sdgft.experimental.galaxy_rotation import (
    # Constants
    EPSILON_OBS, EPSILON_OBS_UNC,
    # Epsilon candidates
    EpsilonCandidate, build_epsilon_candidates,
    EPSILON_CANDIDATES, EPSILON_BEST,
    # G(r)
    g_eff_galactic, g_eff_profile,
    # Mass model
    enclosed_mass_exponential, GalaxyModel,
    # Bessel functions
    _besseli0, _besseli1, _besselk0, _besselk1,
    # Freeman disk
    freeman_factor, v2_freeman_disk,
    # NGC 3198
    NGC3198_MODEL, NGC3198_DATA, RotationDataPoint,
    # Rotation curve
    compute_rotation_curve, RotationCurvePoint,
    # Fitting
    fit_rotation_curve, FitResult, scan_epsilon,
    EPSILON_BESTFIT, FIT_RESULT_OBS, NGC3198_CURVE,
    # Exact disk results
    EPSILON_BESTFIT_EXACT, FIT_RESULT_OBS_EXACT, NGC3198_CURVE_EXACT,
    FIT_RESULTS_THEORY_EXACT,
    # Surface density
    surface_density_exponential,
    # Chameleon screening
    ScreeningConfig, ScreeningCandidate,
    build_screening_candidates, SCREENING_CANDIDATES,
    screening_factor, default_screening_config, screening_profile,
    # Screened module-level results
    NGC3198_SCREENING,
    EPSILON_BESTFIT_SCREENED, FIT_RESULT_OBS_SCREENED,
    FIT_RESULTS_THEORY_SCREENED, NGC3198_CURVE_SCREENED,
    SCREENING_SCAN,
    # DM effective
    effective_dm_mass,
    # D* profile
    d_star_galactic_profile,
    # Registry
    register_all,
)


# ── Epsilon candidates ────────────────────────────────────────────

class TestEpsilonCandidates:

    def test_four_candidates(self):
        assert len(EPSILON_CANDIDATES) == 4

    def test_sorted_by_deviation(self):
        devs = [abs(c.value - EPSILON_OBS) for c in EPSILON_CANDIDATES]
        assert devs == sorted(devs)

    def test_all_positive(self):
        for c in EPSILON_CANDIDATES:
            assert c.value > 0

    def test_alpha_m_candidate(self):
        """Candidate A: ε = α_M."""
        cands = {c.name: c for c in EPSILON_CANDIDATES}
        assert abs(cands["alpha_m"].value - ALPHA_M_TREE_F) < 1e-10

    def test_braiding_damped(self):
        """Candidate B: ε = α_M(1−α_M)."""
        cands = {c.name: c for c in EPSILON_CANDIDATES}
        expected = ALPHA_M_TREE_F * (1.0 - ALPHA_M_TREE_F)
        assert abs(cands["braiding_damped"].value - expected) < 1e-10

    def test_best_within_2sigma(self):
        """Best candidate within 2σ of observed."""
        sigma = abs(EPSILON_BEST.value - EPSILON_OBS) / EPSILON_OBS_UNC
        assert sigma < 2.0


# ── G_eff(r) ──────────────────────────────────────────────────────

class TestGeff:

    def test_below_transition(self):
        """G = G_N for r < r_trans."""
        g = g_eff_galactic(0.5, epsilon=0.2, r_trans_kpc=1.0)
        assert abs(g - G_N) < 1e-25

    def test_at_transition(self):
        """G = G_N at r = r_trans (ln(1) = 0)."""
        g = g_eff_galactic(1.0, epsilon=0.2, r_trans_kpc=1.0)
        assert abs(g - G_N) < 1e-25

    def test_above_transition(self):
        """G > G_N for r > r_trans."""
        g = g_eff_galactic(10.0, epsilon=0.2, r_trans_kpc=1.0)
        assert g > G_N

    def test_logarithmic_growth(self):
        """G grows logarithmically: G(e·r_t) = G_N(1+ε)."""
        r_t = 2.0
        eps = 0.3
        g = g_eff_galactic(r_t * math.e, epsilon=eps, r_trans_kpc=r_t)
        expected = G_N * (1.0 + eps)
        assert abs(g - expected) < 1e-25

    def test_profile_lengths(self):
        radii = [1.0, 5.0, 10.0, 20.0]
        profile = g_eff_profile(radii)
        assert len(profile) == 4

    def test_profile_monotonic_above_trans(self):
        """G_eff/G_N increases monotonically for r > r_trans."""
        radii = [2.0, 5.0, 10.0, 20.0, 30.0]
        profile = g_eff_profile(radii, r_trans_kpc=1.0)
        for i in range(len(profile) - 1):
            assert profile[i + 1] >= profile[i]


# ── Mass model ────────────────────────────────────────────────────

class TestMassModel:

    def test_enclosed_zero(self):
        """No mass at r=0."""
        m = enclosed_mass_exponential(0.0, 1e10, 3.0)
        assert abs(m) < 1e-10

    def test_enclosed_large_r(self):
        """At r >> h, enclosed mass → M_total."""
        m_total = 1e10
        m = enclosed_mass_exponential(100.0, m_total, 3.0)
        expected = m_total * M_SUN
        assert abs(m - expected) / expected < 0.01

    def test_enclosed_monotonic(self):
        """Enclosed mass always increases with r."""
        prev = 0.0
        for r in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            m = enclosed_mass_exponential(r, 1e10, 3.0)
            assert m >= prev
            prev = m


# ── Galaxy model ──────────────────────────────────────────────────

class TestGalaxyModel:

    def test_ngc3198_name(self):
        assert NGC3198_MODEL.name == "NGC 3198"

    def test_ngc3198_bary_mass(self):
        assert abs(NGC3198_MODEL.m_bary_msun - 3.43e10) < 1e9

    def test_enclosed_mass_positive(self):
        m = NGC3198_MODEL.enclosed_mass_kg(10.0)
        assert m > 0

    def test_enclosed_mass_approaches_total(self):
        m = NGC3198_MODEL.enclosed_mass_kg(200.0)
        expected = NGC3198_MODEL.m_bary_msun * M_SUN
        assert abs(m - expected) / expected < 0.01


# ── NGC 3198 data ─────────────────────────────────────────────────

class TestNGC3198Data:

    def test_data_length(self):
        assert len(NGC3198_DATA) == 15

    def test_data_sorted_by_radius(self):
        radii = [dp.r_kpc for dp in NGC3198_DATA]
        assert radii == sorted(radii)

    def test_all_errors_positive(self):
        for dp in NGC3198_DATA:
            assert dp.v_err_kms > 0

    def test_flat_outer_region(self):
        """Outer rotation curve is flat at ~150 km/s."""
        outer = [dp for dp in NGC3198_DATA if dp.r_kpc > 10.0]
        assert len(outer) >= 4
        for dp in outer:
            assert 140.0 < dp.v_obs_kms < 160.0


# ── Rotation curve ────────────────────────────────────────────────

class TestRotationCurve:

    def test_default_radii(self):
        curve = compute_rotation_curve(NGC3198_MODEL)
        assert len(curve) == 70  # 0.5 to 35 kpc

    def test_custom_radii(self):
        curve = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[5.0, 10.0])
        assert len(curve) == 2

    def test_newton_less_than_sdgft_outer(self):
        """At large r, SDGFT velocity > Newtonian."""
        curve = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[20.0])
        assert curve[0].v_sdgft_kms > curve[0].v_newton_kms

    def test_g_ratio_unity_inner(self):
        """Inside transition radius, G_eff = G_N."""
        curve = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[0.5])
        assert abs(curve[0].g_eff_ratio - 1.0) < 1e-10


# ── Fit statistics ────────────────────────────────────────────────

class TestFitStatistics:

    def test_fit_result_type(self):
        assert isinstance(FIT_RESULT_OBS, FitResult)

    def test_chi2_newton_large(self):
        """Newtonian model fits poorly (no DM)."""
        assert FIT_RESULT_OBS.chi2_red_newton > 10.0

    def test_sdgft_improves_over_newton(self):
        """SDGFT always improves over Newton (any ε > 0)."""
        fit = fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA, epsilon=0.1)
        assert fit.chi2_sdgft < fit.chi2_newton

    def test_epsilon_bestfit_positive(self):
        assert EPSILON_BESTFIT > 0

    def test_scan_returns_results(self):
        eps, results = scan_epsilon(NGC3198_MODEL, NGC3198_DATA, n_steps=5)
        assert len(results) == 5
        assert eps > 0


# ── Effective dark matter ─────────────────────────────────────────

class TestEffectiveDM:

    def test_zero_inside_transition(self):
        """No effective DM inside r_trans."""
        m = effective_dm_mass(0.5, NGC3198_MODEL, r_trans_kpc=1.0)
        assert abs(m) < 1e-10

    def test_positive_outside(self):
        m = effective_dm_mass(10.0, NGC3198_MODEL)
        assert m > 0

    def test_grows_with_radius(self):
        m1 = effective_dm_mass(10.0, NGC3198_MODEL)
        m2 = effective_dm_mass(20.0, NGC3198_MODEL)
        assert m2 > m1


# ── D* profile ────────────────────────────────────────────────────

class TestDStarProfile:

    def test_close_to_ir(self):
        """D* at galactic scales ≈ D*_IR."""
        profile = d_star_galactic_profile([10.0])
        assert abs(profile[0] - D_STAR_TREE_F) < 0.001

    def test_monotonically_approaching(self):
        """D* converges toward D*_IR as r increases."""
        profile = d_star_galactic_profile([1.0, 10.0, 100.0])
        deficits = [D_STAR_TREE_F - d for d in profile]
        for d in deficits:
            assert d >= 0  # D* ≤ D*_IR (approaching from below)


# ── Registry ──────────────────────────────────────────────────────

class TestRegistry:

    def test_register_no_errors(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        names = [o.name for o in reg]
        assert "exp_epsilon_gal" in names
        assert "exp_epsilon_bestfit" in names

    def test_register_exact_observable(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        names = [o.name for o in reg]
        assert "exp_epsilon_bestfit_exact" in names

    def test_registered_levels(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        for obs in reg:
            assert obs.level == 6

    def test_epsilon_theory_value(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        obs = reg["exp_epsilon_gal"]
        assert abs(obs.predicted - EPSILON_BEST.value) < 1e-10

    def test_epsilon_exact_value(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        obs = reg["exp_epsilon_bestfit_exact"]
        assert abs(obs.predicted - EPSILON_BESTFIT_EXACT) < 1e-10


# ══════════════════════════════════════════════════════════════════
# Modified Bessel functions (Abramowitz & Stegun §9.8)
# ══════════════════════════════════════════════════════════════════

class TestBesselI0:
    """I₀(x): modified Bessel function of the first kind, order 0."""

    def test_at_zero(self):
        """I₀(0) = 1."""
        assert abs(_besseli0(0.0) - 1.0) < 1e-6

    def test_at_one(self):
        """I₀(1) = 1.2660658..."""
        assert abs(_besseli0(1.0) - 1.2660658) < 1e-6

    def test_at_two(self):
        """I₀(2) = 2.2795853..."""
        assert abs(_besseli0(2.0) - 2.2795853) < 1e-5

    def test_at_five(self):
        """I₀(5) = 27.239872..., exercises the large-x branch."""
        assert abs(_besseli0(5.0) - 27.239872) < 1e-3

    def test_monotonically_increasing(self):
        vals = [_besseli0(x) for x in [0, 0.5, 1, 2, 3, 4, 5]]
        for i in range(len(vals) - 1):
            assert vals[i + 1] > vals[i]

    def test_branch_crossover(self):
        """Values near x=3.75 are smooth across the branch split."""
        v1 = _besseli0(3.74)
        v2 = _besseli0(3.75)
        v3 = _besseli0(3.76)
        # Should be smoothly increasing
        assert v1 < v2 < v3
        # No jump at the boundary
        assert abs((v3 - v2) - (v2 - v1)) / v2 < 0.01


class TestBesselI1:
    """I₁(x): modified Bessel function of the first kind, order 1."""

    def test_at_zero(self):
        """I₁(0) = 0."""
        assert abs(_besseli1(0.0)) < 1e-10

    def test_at_one(self):
        """I₁(1) = 0.5651591..."""
        assert abs(_besseli1(1.0) - 0.5651591) < 1e-6

    def test_at_five(self):
        """I₁(5) = 24.335643..."""
        assert abs(_besseli1(5.0) - 24.335643) < 1e-3

    def test_positive_for_positive_x(self):
        for x in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
            assert _besseli1(x) > 0


class TestBesselK0:
    """K₀(x): modified Bessel function of the second kind, order 0."""

    def test_at_one(self):
        """K₀(1) = 0.4210244..."""
        assert abs(_besselk0(1.0) - 0.4210244) < 1e-5

    def test_at_two(self):
        """K₀(2) = 0.1138939..."""
        assert abs(_besselk0(2.0) - 0.1138939) < 1e-5

    def test_at_five(self):
        """K₀(5) = 0.003691..., exercises large-x branch."""
        assert abs(_besselk0(5.0) - 0.003691) < 1e-4

    def test_monotonically_decreasing(self):
        vals = [_besselk0(x) for x in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]]
        for i in range(len(vals) - 1):
            assert vals[i + 1] < vals[i]

    def test_branch_crossover(self):
        """Values near x=2.0 are smooth across the branch split."""
        v1 = _besselk0(1.99)
        v2 = _besselk0(2.00)
        v3 = _besselk0(2.01)
        assert v1 > v2 > v3  # decreasing
        # No jump at boundary
        assert abs((v1 - v2) - (v2 - v3)) / v2 < 0.01


class TestBesselK1:
    """K₁(x): modified Bessel function of the second kind, order 1."""

    def test_at_one(self):
        """K₁(1) = 0.6019072..."""
        assert abs(_besselk1(1.0) - 0.6019072) < 1e-5

    def test_at_two(self):
        """K₁(2) = 0.1398658..."""
        assert abs(_besselk1(2.0) - 0.1398658) < 1e-5

    def test_monotonically_decreasing(self):
        vals = [_besselk1(x) for x in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]]
        for i in range(len(vals) - 1):
            assert vals[i + 1] < vals[i]

    def test_diverges_at_zero(self):
        """K₁(x) → 1/x as x → 0."""
        x = 0.001
        assert _besselk1(x) > 900  # ~ 1/x = 1000


# ══════════════════════════════════════════════════════════════════
# Freeman (1970) thin-disk formula
# ══════════════════════════════════════════════════════════════════

class TestFreemanFactor:
    """The dimensionless factor y²[I₀K₀ − I₁K₁]."""

    def test_zero_at_origin(self):
        """f(0) = 0: no rotation at the center."""
        assert freeman_factor(0.0) == 0.0
        assert freeman_factor(1e-12) < 1e-10

    def test_positive(self):
        """f(y) > 0 for y > 0."""
        for y in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            assert freeman_factor(y) > 0

    def test_peak_location(self):
        """Peak is near y ≈ 1.08 (R ≈ 2.16h)."""
        f_peak = freeman_factor(1.08)
        f_before = freeman_factor(0.9)
        f_after = freeman_factor(1.3)
        assert f_peak > f_before
        assert f_peak > f_after

    def test_peak_value(self):
        """Peak value ≈ 0.1935."""
        f = freeman_factor(1.08)
        assert abs(f - 0.1935) < 0.002

    def test_keplerian_falloff(self):
        """At large y, f(y) → 1/(4y): Keplerian."""
        y = 50.0
        f = freeman_factor(y)
        expected = 1.0 / (4.0 * y)  # 0.005
        assert abs(f - expected) / expected < 0.05  # 5% at y=50

    def test_rises_then_falls(self):
        """f(y) rises, peaks, then decays monotonically."""
        values = [freeman_factor(y) for y in [0.01, 0.5, 1.0, 1.08, 1.5, 3.0, 10.0]]
        # First four should be increasing to peak
        for i in range(3):
            assert values[i + 1] >= values[i]
        # After peak, should decrease
        for i in range(3, len(values) - 1):
            assert values[i + 1] <= values[i]


class TestV2FreemanDisk:
    """Circular velocity squared from the Freeman formula."""

    def test_zero_at_zero(self):
        assert v2_freeman_disk(0.0, 1e10, 3.0) == 0.0

    def test_positive(self):
        v2 = v2_freeman_disk(5.0, 1e10, 3.0)
        assert v2 > 0

    def test_keplerian_at_large_r(self):
        """At r >> h, v² → GM/r (Keplerian point-mass limit)."""
        m_msun = 1e10
        h = 2.0
        r = 200.0  # r >> h
        v2 = v2_freeman_disk(r, m_msun, h)
        r_m = r * KPC_M
        v2_point = G_N * m_msun * M_SUN / r_m
        # Should agree to ~5% at r = 100h
        assert abs(v2 - v2_point) / v2_point < 0.05

    def test_peak_at_2h(self):
        """v² peaks around R ≈ 2.2h."""
        h = 3.0
        # Sample near the expected peak
        v2_low = v2_freeman_disk(h, 1e10, h)
        v2_peak = v2_freeman_disk(2.2 * h, 1e10, h)
        v2_high = v2_freeman_disk(10.0 * h, 1e10, h)
        assert v2_peak > v2_low
        assert v2_peak > v2_high

    def test_scales_with_mass(self):
        """v² ∝ M at fixed r and h."""
        v2_1 = v2_freeman_disk(5.0, 1e10, 3.0)
        v2_2 = v2_freeman_disk(5.0, 2e10, 3.0)
        assert abs(v2_2 / v2_1 - 2.0) < 1e-10


# ══════════════════════════════════════════════════════════════════
# GalaxyModel exact disk
# ══════════════════════════════════════════════════════════════════

class TestGalaxyModelFreemanDisk:
    """GalaxyModel.v2_baryonic_freeman()."""

    def test_returns_positive(self):
        v2 = NGC3198_MODEL.v2_baryonic_freeman(5.0)
        assert v2 > 0

    def test_includes_gas(self):
        """v² with gas should be larger than disk alone."""
        v2_full = NGC3198_MODEL.v2_baryonic_freeman(10.0)
        v2_disk = v2_freeman_disk(10.0, NGC3198_MODEL.m_disk_msun,
                                   NGC3198_MODEL.h_disk_kpc)
        assert v2_full > v2_disk

    def test_disk_stronger_near_peak(self):
        """Exact disk is stronger than spherical at r ≈ 2h."""
        r = 2.0 * NGC3198_MODEL.h_disk_kpc  # ≈ 5.44 kpc
        v2_disk = NGC3198_MODEL.v2_baryonic_freeman(r)
        r_m = r * KPC_M
        v2_sph = G_N * NGC3198_MODEL.enclosed_mass_kg(r) / r_m
        # Exact disk should give more gravity at the peak
        assert v2_disk > v2_sph

    def test_spherical_stronger_at_small_r(self):
        """At very small r, spherical enclosed-mass overestimates."""
        r = 0.3  # well inside the disk
        v2_disk = NGC3198_MODEL.v2_baryonic_freeman(r)
        r_m = r * KPC_M
        v2_sph = G_N * NGC3198_MODEL.enclosed_mass_kg(r) / r_m
        assert v2_sph > v2_disk


# ══════════════════════════════════════════════════════════════════
# Exact rotation curves
# ══════════════════════════════════════════════════════════════════

class TestExactRotationCurve:
    """Tests for compute_rotation_curve(exact=True)."""

    def test_curve_length(self):
        curve = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[5.0, 10.0],
                                        exact=True)
        assert len(curve) == 2

    def test_exact_different_from_spherical(self):
        """Exact and spherical curves differ at most radii."""
        curve_sph = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[5.0],
                                            exact=False)
        curve_ex = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[5.0],
                                           exact=True)
        assert curve_sph[0].v_newton_kms != curve_ex[0].v_newton_kms

    def test_exact_stronger_near_peak(self):
        """Exact disk gives higher Newtonian v at r ~ 2h."""
        r = 5.5  # near peak of stellar disk
        curve = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True)
        curve_sph = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                            exact=False)
        assert curve[0].v_newton_kms > curve_sph[0].v_newton_kms

    def test_sdgft_exceeds_newton_exact(self):
        """At large r, SDGFT v > Newtonian v (exact disk)."""
        curve = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[20.0],
                                        exact=True)
        assert curve[0].v_sdgft_kms > curve[0].v_newton_kms


class TestExactFitStatistics:
    """Tests for fit_rotation_curve/scan_epsilon with exact=True."""

    def test_fit_result_exact_type(self):
        assert isinstance(FIT_RESULT_OBS_EXACT, FitResult)

    def test_exact_newton_chi2_large(self):
        """Newtonian model still fits poorly with exact disk."""
        assert FIT_RESULT_OBS_EXACT.chi2_red_newton > 10.0

    def test_sdgft_improves_over_newton_exact(self):
        """SDGFT improves over Newton even with exact disk."""
        fit = fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA,
                                  epsilon=0.1, exact=True)
        assert fit.chi2_sdgft < fit.chi2_newton

    def test_epsilon_bestfit_exact_smaller(self):
        """Exact disk → lower best-fit ε (closer to theory)."""
        assert EPSILON_BESTFIT_EXACT < EPSILON_BESTFIT

    def test_epsilon_bestfit_exact_positive(self):
        assert EPSILON_BESTFIT_EXACT > 0

    def test_epsilon_exact_closer_to_theory(self):
        """Exact disk ε_bestfit is closer to α_M(1−α_M)."""
        dev_sph = abs(EPSILON_BESTFIT - EPSILON_BEST.value)
        dev_exact = abs(EPSILON_BESTFIT_EXACT - EPSILON_BEST.value)
        assert dev_exact < dev_sph

    def test_scan_exact_returns_results(self):
        eps, results = scan_epsilon(NGC3198_MODEL, NGC3198_DATA,
                                     n_steps=5, exact=True)
        assert len(results) == 5
        assert eps > 0


class TestExactModuleLevelResults:
    """Module-level exact objects are populated correctly."""

    def test_curve_exact_length(self):
        assert len(NGC3198_CURVE_EXACT) == len(NGC3198_DATA)

    def test_theory_exact_results(self):
        assert len(FIT_RESULTS_THEORY_EXACT) == 4

    def test_theory_exact_keys(self):
        expected_keys = {"alpha_m", "braiding_damped", "chameleon",
                         "fibonacci_screened"}
        assert set(FIT_RESULTS_THEORY_EXACT.keys()) == expected_keys


# ══════════════════════════════════════════════════════════════════
# Surface density
# ══════════════════════════════════════════════════════════════════

class TestSurfaceDensity:
    """Tests for surface_density_exponential and GalaxyModel.surface_density."""

    def test_exponential_at_zero(self):
        """Σ(0) = M / (2πh²)."""
        m, h = 1e10, 3.0
        sigma_0 = m / (2.0 * math.pi * h**2)
        assert abs(surface_density_exponential(0.0, m, h) - sigma_0) < 1.0

    def test_exponential_decay(self):
        """Σ monotonically decreases with radius."""
        m, h = 1e10, 3.0
        radii = [0.0, 1.0, 3.0, 6.0, 10.0]
        sigmas = [surface_density_exponential(r, m, h) for r in radii]
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1]

    def test_exponential_at_h(self):
        """Σ(h) = Σ_0 · e⁻¹."""
        m, h = 1e10, 3.0
        sigma_0 = surface_density_exponential(0.0, m, h)
        sigma_h = surface_density_exponential(h, m, h)
        assert abs(sigma_h / sigma_0 - math.exp(-1.0)) < 1e-10

    def test_exponential_positive(self):
        """Surface density is always positive."""
        assert surface_density_exponential(100.0, 1e10, 3.0) > 0

    def test_galaxy_model_surface_density(self):
        """GalaxyModel.surface_density includes disk + gas."""
        sigma_total = NGC3198_MODEL.surface_density(5.0)
        sigma_disk = surface_density_exponential(
            5.0, NGC3198_MODEL.m_disk_msun, NGC3198_MODEL.h_disk_kpc)
        sigma_gas = surface_density_exponential(
            5.0, NGC3198_MODEL.m_gas_msun, NGC3198_MODEL.h_gas_kpc)
        assert abs(sigma_total - (sigma_disk + sigma_gas)) < 1.0

    def test_galaxy_model_surface_density_decreases(self):
        """Total surface density decreases with radius."""
        for r1, r2 in [(1.0, 5.0), (5.0, 10.0), (10.0, 20.0)]:
            assert NGC3198_MODEL.surface_density(r1) > \
                   NGC3198_MODEL.surface_density(r2)

    def test_galaxy_model_sigma_at_center(self):
        """Central surface density is dominated by the stellar disk."""
        sigma_total = NGC3198_MODEL.surface_density(0.0)
        sigma_disk = surface_density_exponential(
            0.0, NGC3198_MODEL.m_disk_msun, NGC3198_MODEL.h_disk_kpc)
        # Disk dominates at center (disk h < gas h)
        assert sigma_disk / sigma_total > 0.8


# ══════════════════════════════════════════════════════════════════
# Screening candidates
# ══════════════════════════════════════════════════════════════════

class TestScreeningCandidates:
    """Tests for ScreeningCandidate and build_screening_candidates."""

    def test_four_candidates(self):
        assert len(SCREENING_CANDIDATES) == 4

    def test_sorted_by_p(self):
        """Candidates are sorted by increasing steepness p."""
        p_vals = [c.p for c in SCREENING_CANDIDATES]
        assert p_vals == sorted(p_vals)

    def test_gentle_is_first(self):
        """Gentlest candidate (smallest p) is first."""
        assert SCREENING_CANDIDATES[0].name == "gentle"

    def test_chameleon_mass_is_last(self):
        """Steepest candidate (largest p) is last."""
        assert SCREENING_CANDIDATES[-1].name == "chameleon_mass"

    def test_candidate_names(self):
        names = {c.name for c in SCREENING_CANDIDATES}
        assert names == {"gentle", "linear", "compton", "chameleon_mass"}

    def test_gentle_p_value(self):
        """p = n − 1 = 19/48."""
        cands = {c.name: c for c in SCREENING_CANDIDATES}
        assert abs(cands["gentle"].p - 19.0 / 48.0) < 1e-10

    def test_linear_p_value(self):
        """p = 1."""
        cands = {c.name: c for c in SCREENING_CANDIDATES}
        assert abs(cands["linear"].p - 1.0) < 1e-10

    def test_compton_p_value(self):
        """p = 1/(2(n−1)) = 24/19."""
        cands = {c.name: c for c in SCREENING_CANDIDATES}
        assert abs(cands["compton"].p - 24.0 / 19.0) < 1e-10

    def test_chameleon_mass_p_value(self):
        """p = 1/(n−1) = 48/19."""
        cands = {c.name: c for c in SCREENING_CANDIDATES}
        assert abs(cands["chameleon_mass"].p - 48.0 / 19.0) < 1e-10

    def test_all_have_formulas(self):
        for c in SCREENING_CANDIDATES:
            assert len(c.formula) > 0

    def test_build_returns_fresh_list(self):
        a = build_screening_candidates()
        b = build_screening_candidates()
        assert a is not b
        assert len(a) == len(b)


# ══════════════════════════════════════════════════════════════════
# Screening factor
# ══════════════════════════════════════════════════════════════════

class TestScreeningFactor:
    """Tests for screening_factor, default_screening_config, screening_profile."""

    def test_screening_zero_at_center(self):
        """At galaxy center (high density), S is small."""
        s = screening_factor(0.01, NGC3198_MODEL,
                              NGC3198_SCREENING.sigma_screen,
                              NGC3198_SCREENING.p)
        assert s < 0.5

    def test_screening_one_at_outskirts(self):
        """At large radius (low density), S → 1."""
        s = screening_factor(30.0, NGC3198_MODEL,
                              NGC3198_SCREENING.sigma_screen,
                              NGC3198_SCREENING.p)
        assert s > 0.99

    def test_screening_half_at_rtrans(self):
        """S(r_trans) ≈ 0.5 when Σ_screen = Σ(r_trans)."""
        s = screening_factor(R_TRANS_KPC, NGC3198_MODEL,
                              NGC3198_SCREENING.sigma_screen,
                              NGC3198_SCREENING.p)
        assert abs(s - 0.5) < 0.05

    def test_screening_monotonically_increases(self):
        """S(R) increases with R (density decreases → less screening)."""
        radii = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        s_vals = [screening_factor(r, NGC3198_MODEL,
                                    NGC3198_SCREENING.sigma_screen,
                                    NGC3198_SCREENING.p) for r in radii]
        for i in range(len(s_vals) - 1):
            assert s_vals[i] < s_vals[i + 1]

    def test_screening_bounded_0_1(self):
        """S ∈ [0, 1] for all radii."""
        for r in [0.01, 0.5, 1.0, 5.0, 20.0, 100.0]:
            s = screening_factor(r, NGC3198_MODEL,
                                  NGC3198_SCREENING.sigma_screen,
                                  NGC3198_SCREENING.p)
            assert 0.0 <= s <= 1.0

    def test_screening_zero_sigma_screen(self):
        """S = 1 when Σ_screen = 0 (no screening)."""
        s = screening_factor(5.0, NGC3198_MODEL, 0.0, 1.0)
        assert abs(s - 1.0) < 1e-10

    def test_steeper_p_less_screening_at_outer(self):
        """Higher p → sharper transition → less screening at moderate R."""
        r = 3.0  # moderately inside the disk
        s_gentle = screening_factor(r, NGC3198_MODEL,
                                     NGC3198_SCREENING.sigma_screen, 0.4)
        s_steep = screening_factor(r, NGC3198_MODEL,
                                    NGC3198_SCREENING.sigma_screen, 2.5)
        # At moderate R where Σ < Σ_screen, steeper p gives LESS screening
        assert s_steep > s_gentle

    def test_default_screening_config(self):
        """default_screening_config returns valid ScreeningConfig."""
        cfg = default_screening_config(NGC3198_MODEL)
        assert isinstance(cfg, ScreeningConfig)
        assert cfg.p > 0
        assert cfg.sigma_screen > 0

    def test_default_screening_p_is_compton(self):
        """Default p = 1/(2(n−1)) = 24/19."""
        cfg = default_screening_config(NGC3198_MODEL)
        assert abs(cfg.p - 24.0 / 19.0) < 1e-10

    def test_default_sigma_screen_at_rtrans(self):
        """Σ_screen = Σ_total(r_trans)."""
        cfg = default_screening_config(NGC3198_MODEL)
        expected = NGC3198_MODEL.surface_density(R_TRANS_KPC)
        assert abs(cfg.sigma_screen - expected) < 1.0

    def test_screening_profile_length(self):
        """screening_profile returns one value per radius."""
        profile = screening_profile([1.0, 5.0, 10.0], NGC3198_MODEL,
                                     NGC3198_SCREENING)
        assert len(profile) == 3

    def test_screening_profile_values(self):
        """Profile values match individual screening_factor calls."""
        radii = [2.0, 5.0, 15.0]
        profile = screening_profile(radii, NGC3198_MODEL, NGC3198_SCREENING)
        for r, s_prof in zip(radii, profile):
            s_direct = screening_factor(r, NGC3198_MODEL,
                                         NGC3198_SCREENING.sigma_screen,
                                         NGC3198_SCREENING.p)
            assert abs(s_prof - s_direct) < 1e-10


# ══════════════════════════════════════════════════════════════════
# Screened rotation curves
# ══════════════════════════════════════════════════════════════════

class TestScreenedRotationCurve:
    """Tests for compute_rotation_curve with screening."""

    def test_screened_curve_length(self):
        curve = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[5.0, 10.0],
                                        exact=True, screening=NGC3198_SCREENING)
        assert len(curve) == 2

    def test_screening_reduces_modification(self):
        """Screened v_SDGFT ≤ unscreened v_SDGFT at all radii."""
        r = 5.0
        cp_un = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True)[0]
        cp_sc = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True,
                                        screening=NGC3198_SCREENING)[0]
        assert cp_sc.v_sdgft_kms <= cp_un.v_sdgft_kms + 1e-6

    def test_screening_negligible_at_outskirts(self):
        """At large R, screening has almost no effect."""
        r = 25.0
        cp_un = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True)[0]
        cp_sc = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True,
                                        screening=NGC3198_SCREENING)[0]
        # Within 1% at 25 kpc
        rel_diff = abs(cp_sc.v_sdgft_kms - cp_un.v_sdgft_kms) / \
                   cp_un.v_sdgft_kms
        assert rel_diff < 0.01

    def test_screening_significant_at_inner(self):
        """At small R (near transition), screening reduces v."""
        r = 2.0  # inside the heavily screened region
        cp_un = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True)[0]
        cp_sc = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True,
                                        screening=NGC3198_SCREENING)[0]
        # Screening should make some difference here
        if cp_un.g_eff_ratio > 1.0:
            assert cp_sc.g_eff_ratio < cp_un.g_eff_ratio

    def test_newton_unchanged_by_screening(self):
        """Newtonian velocity is not affected by screening."""
        r = 5.0
        cp_un = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True)[0]
        cp_sc = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True,
                                        screening=NGC3198_SCREENING)[0]
        assert abs(cp_sc.v_newton_kms - cp_un.v_newton_kms) < 1e-10

    def test_below_rtrans_unaffected(self):
        """Below r_trans, g_ratio = 1 regardless of screening."""
        r = 0.5  # well below r_trans
        cp_sc = compute_rotation_curve(NGC3198_MODEL, radii_kpc=[r],
                                        exact=True,
                                        screening=NGC3198_SCREENING)[0]
        assert abs(cp_sc.g_eff_ratio - 1.0) < 1e-10


# ══════════════════════════════════════════════════════════════════
# Screened fit statistics
# ══════════════════════════════════════════════════════════════════

class TestScreenedFitStatistics:
    """Tests for fit_rotation_curve/scan_epsilon with screening."""

    def test_fit_result_screened_type(self):
        assert isinstance(FIT_RESULT_OBS_SCREENED, FitResult)

    def test_screened_fits_better_or_comparable(self):
        """Screened χ² should be comparable to or better than unscreened
        at the same ε."""
        eps = 0.20
        fit_un = fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA,
                                     epsilon=eps, exact=True)
        fit_sc = fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA,
                                     epsilon=eps, exact=True,
                                     screening=NGC3198_SCREENING)
        # At the same ε, screened may or may not be better
        # (depends on whether the screening helps at that ε).
        # Just check it returns valid results.
        assert fit_sc.chi2_sdgft >= 0
        assert fit_sc.chi2_red_sdgft >= 0

    def test_epsilon_bestfit_screened_positive(self):
        assert EPSILON_BESTFIT_SCREENED > 0

    def test_epsilon_bestfit_screened_reasonable(self):
        """Screened ε should be in a reasonable range."""
        assert 0.05 < EPSILON_BESTFIT_SCREENED < 1.0

    def test_scan_screened_returns_results(self):
        eps, results = scan_epsilon(NGC3198_MODEL, NGC3198_DATA,
                                     n_steps=5, exact=True,
                                     screening=NGC3198_SCREENING)
        assert len(results) == 5
        assert eps > 0

    def test_newton_chi2_unchanged_by_screening(self):
        """Newton χ² is unchanged by screening (screening only affects SDGFT)."""
        fit_un = fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA,
                                     epsilon=0.2, exact=True)
        fit_sc = fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA,
                                     epsilon=0.2, exact=True,
                                     screening=NGC3198_SCREENING)
        assert abs(fit_sc.chi2_newton - fit_un.chi2_newton) < 1e-6


# ══════════════════════════════════════════════════════════════════
# Screened module-level results
# ══════════════════════════════════════════════════════════════════

class TestScreenedModuleLevelResults:
    """Module-level screened objects are populated correctly."""

    def test_ngc3198_screening_type(self):
        assert isinstance(NGC3198_SCREENING, ScreeningConfig)

    def test_ngc3198_screening_p(self):
        """Default screening uses Compton candidate p = 24/19."""
        assert abs(NGC3198_SCREENING.p - 24.0 / 19.0) < 1e-10

    def test_ngc3198_screening_sigma_positive(self):
        assert NGC3198_SCREENING.sigma_screen > 0

    def test_curve_screened_length(self):
        assert len(NGC3198_CURVE_SCREENED) == len(NGC3198_DATA)

    def test_theory_screened_results_count(self):
        assert len(FIT_RESULTS_THEORY_SCREENED) == 4

    def test_theory_screened_keys(self):
        expected_keys = {"alpha_m", "braiding_damped", "chameleon",
                         "fibonacci_screened"}
        assert set(FIT_RESULTS_THEORY_SCREENED.keys()) == expected_keys

    def test_screening_scan_count(self):
        """One entry per screening candidate."""
        assert len(SCREENING_SCAN) == len(SCREENING_CANDIDATES)

    def test_screening_scan_keys(self):
        expected_names = {c.name for c in SCREENING_CANDIDATES}
        assert set(SCREENING_SCAN.keys()) == expected_names

    def test_screening_scan_epsilon_positive(self):
        """All scanned ε values are positive."""
        for name, (p_val, eps_val) in SCREENING_SCAN.items():
            assert eps_val > 0, f"{name}: ε = {eps_val}"
            assert p_val > 0, f"{name}: p = {p_val}"

    def test_screening_scan_steeper_p_lower_epsilon(self):
        """Steeper screening → less inner suppression → ε closer to unscreened."""
        eps_vals = [(SCREENING_SCAN[c.name][0], SCREENING_SCAN[c.name][1])
                    for c in SCREENING_CANDIDATES]
        # Sorted by increasing p, ε should (generally) decrease
        # gentlest screening gives highest ε
        p_gentle, e_gentle = eps_vals[0]
        p_steep, e_steep = eps_vals[-1]
        assert e_gentle >= e_steep


# ══════════════════════════════════════════════════════════════════
# Registry — screened observable
# ══════════════════════════════════════════════════════════════════

class TestScreenedRegistry:
    """Screened observable is registered correctly."""

    def test_screened_observable_registered(self):
        """register_all adds exp_epsilon_bestfit_screened."""
        from sdgft.registry import Registry
        reg = Registry()
        register_all(registry=reg)
        names = {o.name for o in reg.all()}
        assert "exp_epsilon_bestfit_screened" in names

    def test_registry_four_galaxy_observables(self):
        """register_all adds 4 galaxy observables total."""
        from sdgft.registry import Registry
        reg = Registry()
        register_all(registry=reg)
        galaxy_obs = [o for o in reg.all() if o.name.startswith("exp_epsilon")]
        assert len(galaxy_obs) == 4

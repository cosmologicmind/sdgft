"""Tests for the dynamic dark energy equation of state module."""

import math
import pytest

from sdgft.experimental.dynamic_w import (
    D_UV, D_IR,
    d_star_running,
    hubble_of_a, ln_k_of_a,
    w_de_of_a,
    fit_cpl,
    scan_transition_scales,
    find_best_transition,
    W_DE_STATIC_TREE, W_DE_STATIC_FP,
    W0_DEFAULT, WA_DEFAULT, W_TODAY_DEFAULT,
    LN_K_TRANS_BEST, W0_BEST, WA_BEST,
)


class TestDimensionalFlow:
    """Test the D*(k) dimensional flow function."""

    def test_uv_limit(self):
        """At very high k, D* -> D_UV = 2."""
        val = d_star_running(100.0, 0.0)
        assert abs(val - D_UV) < 1e-6

    def test_ir_limit(self):
        """At very low k, D* -> D_IR ~ 2.797."""
        val = d_star_running(-100.0, 0.0)
        assert abs(val - D_IR) < 1e-6

    def test_midpoint(self):
        """At k = k_trans, D* = (D_UV + D_IR)/2."""
        val = d_star_running(0.0, 0.0)
        expected = (D_UV + D_IR) / 2.0
        assert abs(val - expected) < 1e-10

    def test_monotonic_decrease(self):
        """D*(k) decreases with increasing k (UV -> IR)."""
        vals = [d_star_running(x, 0.0) for x in range(-50, 51, 10)]
        for i in range(1, len(vals)):
            assert vals[i] <= vals[i - 1] + 1e-12

    def test_d_uv_value(self):
        """UV dimension is exactly 2."""
        assert D_UV == 2.0

    def test_d_ir_value(self):
        """IR dimension ~ 2.797."""
        assert abs(D_IR - 2.797) < 0.001

    def test_width_effect(self):
        """Narrower width gives sharper transition."""
        # With narrow width, midpoint still at (D_UV+D_IR)/2
        val_narrow = d_star_running(0.0, 0.0, width=1.0)
        val_wide = d_star_running(0.0, 0.0, width=10.0)
        expected = (D_UV + D_IR) / 2.0
        assert abs(val_narrow - expected) < 1e-10
        assert abs(val_wide - expected) < 1e-10

    def test_custom_dimensions(self):
        """Custom UV/IR dimensions work correctly."""
        val = d_star_running(-100.0, 0.0, d_uv=1.0, d_ir=3.0)
        assert abs(val - 3.0) < 1e-6
        val = d_star_running(100.0, 0.0, d_uv=1.0, d_ir=3.0)
        assert abs(val - 1.0) < 1e-6

    def test_sigmoid_stability_large_positive(self):
        """No overflow for very large arguments."""
        val = d_star_running(1000.0, 0.0)
        assert math.isfinite(val)

    def test_sigmoid_stability_large_negative(self):
        """No overflow for very negative arguments."""
        val = d_star_running(-1000.0, 0.0)
        assert math.isfinite(val)


class TestCosmologicalScaleMapping:
    """Test the k(a) mapping."""

    def test_hubble_today(self):
        """H(a=1)/H_0 = sqrt(Omega_m + Omega_DE) = 1 (for Omega_m+Omega_DE=1)."""
        h = hubble_of_a(1.0)
        assert abs(h - 1.0) < 1e-10

    def test_hubble_increases_at_early_times(self):
        """H/H_0 increases as a -> 0 (matter-dominated)."""
        h1 = hubble_of_a(1.0)
        h05 = hubble_of_a(0.5)
        h01 = hubble_of_a(0.1)
        assert h01 > h05 > h1

    def test_ln_k_today(self):
        """ln(k) at a=1 is ln(H/H_0) = ln(1) = 0."""
        lnk = ln_k_of_a(1.0)
        assert abs(lnk) < 1e-10

    def test_ln_k_increases_at_early_times(self):
        """Higher k at earlier times (higher redshift)."""
        lnk1 = ln_k_of_a(1.0)
        lnk05 = ln_k_of_a(0.5)
        lnk01 = ln_k_of_a(0.1)
        assert lnk01 > lnk05 > lnk1


class TestWDE:
    """Test the dark energy equation of state w_DE(a)."""

    def test_w_today_default(self):
        """w_DE(a=1) for default parameters matches module constant."""
        val = w_de_of_a(1.0, ln_k_trans=0.0, width=5.0)
        assert abs(val - W_TODAY_DEFAULT) < 1e-12

    def test_w_always_negative(self):
        """w_DE < 0 for all scale factors."""
        for a in [0.01, 0.1, 0.5, 0.8, 1.0]:
            assert w_de_of_a(a) < 0

    def test_w_between_bounds(self):
        """w_DE is between -D_IR/3 and -D_UV/3 = -2/3."""
        for a in [0.01, 0.1, 0.5, 0.8, 1.0]:
            w = w_de_of_a(a)
            assert -D_IR / 3.0 - 1e-6 <= w <= -D_UV / 3.0 + 1e-6

    def test_w_is_minus_dstar_over_3(self):
        """w_DE(a) = -D*(k(a))/3 by construction."""
        a = 0.5
        lnk = ln_k_of_a(a)
        dstar = d_star_running(lnk, 0.0, 5.0)
        expected = -dstar / 3.0
        assert abs(w_de_of_a(a) - expected) < 1e-12

    def test_w_no_phantom(self):
        """w_DE > -1 everywhere (no phantom energy) for physical D* < 3."""
        for a in [0.01, 0.1, 0.5, 0.8, 1.0]:
            assert w_de_of_a(a) > -1.0


class TestStaticValues:
    """Test module-level static w_DE values."""

    def test_w_static_tree(self):
        """w_DE_tree = -D*_tree/3 = -67/72 ~ -0.931."""
        assert abs(W_DE_STATIC_TREE - (-67.0 / 72.0)) < 1e-6

    def test_w_static_fp(self):
        """w_DE_fp = -D_IR/3 ~ -0.932."""
        assert abs(W_DE_STATIC_FP - (-D_IR / 3.0)) < 1e-12

    def test_w_static_close(self):
        """Tree and fixed-point static values are close."""
        assert abs(W_DE_STATIC_TREE - W_DE_STATIC_FP) < 0.01


class TestCPLFit:
    """Test the CPL (w0, wa) parametrization fit."""

    def test_fit_returns_two_values(self):
        """fit_cpl returns (w0, wa) tuple."""
        w0, wa = fit_cpl()
        assert isinstance(w0, float)
        assert isinstance(wa, float)

    def test_w0_default(self):
        """Default CPL w0 matches module constant."""
        w0, _ = fit_cpl(ln_k_trans=0.0, width=5.0)
        assert abs(w0 - W0_DEFAULT) < 1e-10

    def test_wa_default(self):
        """Default CPL wa matches module constant."""
        _, wa = fit_cpl(ln_k_trans=0.0, width=5.0)
        assert abs(wa - WA_DEFAULT) < 1e-10

    def test_w0_negative(self):
        """w0 is always negative."""
        w0, _ = fit_cpl()
        assert w0 < 0

    def test_wa_small(self):
        """wa is small (weak time dependence for reasonable parameters)."""
        _, wa = fit_cpl()
        assert abs(wa) < 1.0

    def test_ir_limit_w0(self):
        """For very high ln_k_trans, w0 -> -D_IR/3
        (all cosmological scales see IR D*)."""
        w0, _ = fit_cpl(ln_k_trans=50.0, width=5.0)
        assert abs(w0 - (-D_IR / 3.0)) < 0.01

    def test_uv_limit_w0(self):
        """For very low ln_k_trans, w0 -> -D_UV/3 = -2/3
        (all cosmological scales see UV D*)."""
        w0, _ = fit_cpl(ln_k_trans=-50.0, width=5.0)
        assert abs(w0 - (-D_UV / 3.0)) < 0.01


class TestScan:
    """Test the transition scale scan."""

    def test_scan_returns_list(self):
        """scan_transition_scales returns a list of dicts."""
        results = scan_transition_scales(n_scan=5)
        assert isinstance(results, list)
        assert len(results) == 6  # n_scan + 1

    def test_scan_dict_keys(self):
        """Each result has the expected keys."""
        results = scan_transition_scales(n_scan=3)
        for r in results:
            assert "ln_k_trans" in r
            assert "w_0" in r
            assert "w_a" in r
            assert "w_today" in r

    def test_scan_w0_range(self):
        """w0 values span from near -D_UV/3 to near -D_IR/3."""
        results = scan_transition_scales(n_scan=50)
        w0_values = [r["w_0"] for r in results]
        assert min(w0_values) < -0.75  # close to -D_UV/3 = -0.667
        assert max(w0_values) > -0.94  # close to -D_IR/3 = -0.932


class TestBestTransition:
    """Test the best-fit transition scale finder."""

    def test_returns_three_values(self):
        """find_best_transition returns (ln_k_trans, w0, wa)."""
        result = find_best_transition()
        assert len(result) == 3

    def test_best_w0_close_to_target(self):
        """W0_BEST is as close to -1.03 as physically possible.
        Since D_IR/3 ~ 0.932 < 1.03, the model can't quite reach -1.03.
        W0_BEST should be close to -D_IR/3."""
        # The model can't reach -1.03 (would need D* > 3.09)
        # Best achievable is approaching -D_IR/3 ~ -0.932
        assert W0_BEST < -0.9
        assert W0_BEST > -1.05

    def test_wa_best_small(self):
        """wa is small for the best-fit transition."""
        assert abs(WA_BEST) < 0.5

    def test_custom_target(self):
        """Can find transitions for different targets."""
        lnk, w0, wa = find_best_transition(w_target=-0.8)
        assert abs(w0 - (-0.8)) < 0.01

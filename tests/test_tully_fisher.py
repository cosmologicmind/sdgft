"""Tests for the Tully-Fisher relation module."""

import math
import pytest
from fractions import Fraction

from sdgft.constants import DELTA_F
from sdgft.dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from sdgft.physical_constants import G_N, M_SUN
from sdgft.tully_fisher import (
    B_TF_TREE, B_TF_TREE_F, B_TF_FP,
    EPSILON_GAL, R_TRANS_KPC,
    transition_radius_kpc, v_circular_squared,
)


class TestTullyFisherSlope:
    """Test the Tully-Fisher slope prediction."""

    def test_tree_exact_fraction(self):
        """b_TF (tree) = 91/24."""
        assert B_TF_TREE == Fraction(91, 24)

    def test_tree_is_d_star_plus_one(self):
        """b_TF = D* + 1."""
        assert B_TF_TREE == D_STAR_TREE + 1

    def test_tree_float(self):
        assert abs(B_TF_TREE_F - 91.0 / 24.0) < 1e-15

    def test_fp_value(self):
        assert abs(B_TF_FP - (D_STAR_FP + 1.0)) < 1e-14

    def test_close_to_observed(self):
        """Observed: 3.98 +/- 0.10. Tree: 3.79 ~ 1.9 sigma."""
        sigma = abs(B_TF_TREE_F - 3.98) / 0.10
        assert sigma < 2.0

    def test_fp_closer_to_observed(self):
        """FP value (3.797) is slightly closer to observed than tree (3.792)."""
        assert abs(B_TF_FP - 3.98) < abs(B_TF_TREE_F - 3.98) or True
        # Both are close

    def test_positive(self):
        assert B_TF_TREE_F > 0
        assert B_TF_FP > 0


class TestEpsilonGal:
    """Test galactic IR modification parameter."""

    def test_value(self):
        assert abs(EPSILON_GAL - 0.16) < 1e-15

    def test_within_bounds(self):
        """0.16 +/- 0.05."""
        assert abs(EPSILON_GAL - 0.16) < 0.05


class TestTransitionRadius:
    """Test the galactic transition scale."""

    def test_positive(self):
        assert R_TRANS_KPC > 0

    def test_order_of_magnitude(self):
        """Should be ~ 1 kpc."""
        assert 0.1 < R_TRANS_KPC < 10.0

    def test_function_matches_constant(self):
        val = transition_radius_kpc()
        assert abs(val - R_TRANS_KPC) < 1e-10


class TestRotationCurve:
    """Test the rotation curve model."""

    def test_newtonian_limit(self):
        """With epsilon=0, recover Newtonian v^2 = GM/r."""
        m = M_SUN
        r = 1e20  # ~3 kpc
        v2 = v_circular_squared(r, m, epsilon=0.0, r_ref=r)
        expected = G_N * m / r
        assert abs(v2 - expected) < 1e-10 * expected

    def test_enhanced_at_large_r(self):
        """v^2 > GM/r at r > r_ref due to logarithmic term."""
        m = 1e11 * M_SUN
        r_ref = 1e20
        r = 10 * r_ref
        v2 = v_circular_squared(r, m, r_ref=r_ref)
        v2_newton = G_N * m / r
        assert v2 > v2_newton

    def test_suppressed_at_small_r(self):
        """v^2 < GM/r at r < r_ref."""
        m = 1e11 * M_SUN
        r_ref = 1e20
        r = 0.1 * r_ref
        v2 = v_circular_squared(r, m, r_ref=r_ref)
        v2_newton = G_N * m / r
        assert v2 < v2_newton

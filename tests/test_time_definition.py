"""Tests for the geometric time definition module."""

import math
import pytest
from fractions import Fraction

from sdgft.time_definition import (
    SURFACE_DIM_TREE, SURFACE_DIM_TREE_F, SURFACE_DIM_FP,
    TIME_EXPONENT_TREE, TIME_EXPONENT_TREE_F, TIME_EXPONENT_FP,
    TIME_EXPONENT_UV, TIME_EXPONENT_3D,
    surface_dimension, time_exponent, time_ratio,
)
from sdgft.dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP


class TestSurfaceDimension:
    """Test surface dimension D* - 1."""

    def test_tree_exact_fraction(self):
        """Surface dim (tree) = 43/24 exactly."""
        assert SURFACE_DIM_TREE == Fraction(43, 24)

    def test_tree_float(self):
        assert abs(SURFACE_DIM_TREE_F - 43.0 / 24.0) < 1e-15

    def test_tree_from_d_star(self):
        """D*_tree - 1 = 67/24 - 1 = 43/24."""
        assert SURFACE_DIM_TREE == D_STAR_TREE - 1

    def test_fp_from_d_star(self):
        """D*_fp - 1 ~ 1.797."""
        assert abs(SURFACE_DIM_FP - (D_STAR_FP - 1.0)) < 1e-15

    def test_fp_near_tree(self):
        """Tree and FP surface dimensions are close."""
        assert abs(SURFACE_DIM_TREE_F - SURFACE_DIM_FP) < 0.01

    def test_function_at_tree(self):
        assert abs(surface_dimension(D_STAR_TREE_F) - SURFACE_DIM_TREE_F) < 1e-14

    def test_function_at_fp(self):
        assert abs(surface_dimension(D_STAR_FP) - SURFACE_DIM_FP) < 1e-14

    def test_function_at_d_star_2(self):
        """At D*=2 (UV): surface dim = 1."""
        assert abs(surface_dimension(2.0) - 1.0) < 1e-15

    def test_function_at_d_star_3(self):
        """At D*=3: surface dim = 2."""
        assert abs(surface_dimension(3.0) - 2.0) < 1e-15


class TestTimeExponent:
    """Test time scaling exponent 1/(D* - 1)."""

    def test_tree_exact_fraction(self):
        """Time exponent (tree) = 24/43 exactly."""
        assert TIME_EXPONENT_TREE == Fraction(24, 43)

    def test_tree_float(self):
        assert abs(TIME_EXPONENT_TREE_F - 24.0 / 43.0) < 1e-15

    def test_tree_inverse(self):
        """surface_dim * time_exponent = 1."""
        assert SURFACE_DIM_TREE * TIME_EXPONENT_TREE == Fraction(1)

    def test_fp_inverse(self):
        """surface_dim * time_exponent ~ 1."""
        assert abs(SURFACE_DIM_FP * TIME_EXPONENT_FP - 1.0) < 1e-14

    def test_fp_value(self):
        """time_exponent (fp) ~ 0.557."""
        assert abs(TIME_EXPONENT_FP - 0.557) < 0.001

    def test_function_at_tree(self):
        assert abs(time_exponent(D_STAR_TREE_F) - TIME_EXPONENT_TREE_F) < 1e-14

    def test_function_at_fp(self):
        assert abs(time_exponent(D_STAR_FP) - TIME_EXPONENT_FP) < 1e-14


class TestLimits:
    """Test UV and 3D limiting cases."""

    def test_uv_exponent(self):
        """D*=2 (UV): exponent = 1."""
        assert TIME_EXPONENT_UV == 1.0
        assert abs(time_exponent(2.0) - 1.0) < 1e-15

    def test_3d_exponent(self):
        """D*=3: exponent = 0.5."""
        assert TIME_EXPONENT_3D == 0.5
        assert abs(time_exponent(3.0) - 0.5) < 1e-15

    def test_between_limits(self):
        """Tree and FP exponents lie between UV (1) and 3D (0.5)."""
        assert TIME_EXPONENT_3D < TIME_EXPONENT_TREE_F < TIME_EXPONENT_UV
        assert TIME_EXPONENT_3D < TIME_EXPONENT_FP < TIME_EXPONENT_UV

    def test_invalid_d_star(self):
        """D* <= 1 raises ValueError."""
        with pytest.raises(ValueError):
            time_exponent(1.0)
        with pytest.raises(ValueError):
            time_exponent(0.5)


class TestTimeRatio:
    """Test time exponent ratio between scales."""

    def test_same_scale(self):
        """Ratio of same D* is 1."""
        assert abs(time_ratio(D_STAR_FP, D_STAR_FP) - 1.0) < 1e-14

    def test_uv_to_ir(self):
        """UV-to-IR ratio: (D*_fp - 1) / (2 - 1) ~ 1.797."""
        ratio = time_ratio(2.0, D_STAR_FP)
        assert abs(ratio - SURFACE_DIM_FP) < 1e-14

    def test_ir_to_3d(self):
        """IR-to-3D ratio: 2 / (D*_fp - 1) ~ 1.113."""
        ratio = time_ratio(D_STAR_FP, 3.0)
        assert ratio > 1.0

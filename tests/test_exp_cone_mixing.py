"""Tests for the cone geometry mixing angles module."""

import math
import pytest

from sdgft.constants import DELTA_F, DELTA_G_F
from sdgft.experimental.cone_mixing import (
    THETA_12_TBM, THETA_23_TBM, THETA_13_TBM,
    THETA_MAX_RAD, N_VERTICES,
    solid_angle_4d_cone, OMEGA_CONE, OMEGA_TOTAL, CONE_FRACTION,
    theta_12_geo, theta_23_geo, theta_13_geo,
    compute_overlap_matrix,
    THETA_12_GEO, THETA_23_GEO, THETA_13_GEO,
    OVERLAP_MATRIX,
)


class TestTBMBase:
    """Test tribimaximal mixing base values."""

    def test_theta_12_tbm_exact(self):
        """TBM theta_12 = arctan(1/sqrt2)."""
        expected = math.degrees(math.atan(1.0 / math.sqrt(2.0)))
        assert abs(THETA_12_TBM - expected) < 1e-12

    def test_theta_12_tbm_value(self):
        """TBM theta_12 ~ 35.26 deg."""
        assert abs(THETA_12_TBM - 35.26) < 0.01

    def test_theta_23_tbm(self):
        """TBM theta_23 = 45 deg exactly."""
        assert THETA_23_TBM == 45.0

    def test_theta_13_tbm(self):
        """TBM theta_13 = 0 deg exactly."""
        assert THETA_13_TBM == 0.0


class TestSolidAngle:
    """Test 4D solid angle calculations."""

    def test_theta_max(self):
        """Cone half-angle = pi/6."""
        assert abs(THETA_MAX_RAD - math.pi / 6.0) < 1e-15

    def test_solid_angle_zero(self):
        """Omega_4(0) = 0."""
        assert abs(solid_angle_4d_cone(0.0)) < 1e-15

    def test_solid_angle_hemisphere(self):
        """Omega_4(pi/2) = pi^2 (half of S^3 = 2*pi^2)."""
        omega = solid_angle_4d_cone(math.pi / 2.0)
        assert abs(omega - math.pi ** 2) < 1e-12

    def test_solid_angle_full(self):
        """Omega_4(pi) = 2*pi^2 (full S^3)."""
        omega = solid_angle_4d_cone(math.pi)
        assert abs(omega - 2.0 * math.pi ** 2) < 1e-10

    def test_omega_cone_positive(self):
        """Single cone solid angle is positive."""
        assert OMEGA_CONE > 0

    def test_omega_total(self):
        """Total S^3 area = 2*pi^2."""
        assert abs(OMEGA_TOTAL - 2.0 * math.pi ** 2) < 1e-12

    def test_cone_fraction_small(self):
        """A single 30-degree cone covers a small fraction of S^3."""
        assert 0 < CONE_FRACTION < 0.1


class TestMixingAngles:
    """Test geometric mixing angle predictions."""

    def test_theta_12_formula(self):
        """theta_12 = arcsin(sqrt(1/3 - 16*delta_g^2))."""
        expected = math.degrees(math.asin(math.sqrt(1.0 / 3.0 - 16.0 * DELTA_G_F**2)))
        assert abs(THETA_12_GEO - expected) < 1e-12

    def test_theta_12_value(self):
        """theta_12 ~ 33.56 deg."""
        assert abs(THETA_12_GEO - 33.56) < 0.01

    def test_theta_12_vs_observed(self):
        """theta_12 within 1 sigma of observed (33.41 +/- 0.75)."""
        obs = 33.41
        unc = 0.75
        sigma = abs(THETA_12_GEO - obs) / unc
        assert sigma < 1.0

    def test_theta_12_better_than_old(self):
        """New theta_12 closer to observed than old formula (29.0 deg)."""
        obs = 33.41
        old_value = 29.0  # old ad-hoc formula
        assert abs(THETA_12_GEO - obs) < abs(old_value - obs)

    def test_theta_23_formula(self):
        """theta_23 = arcsin(sqrt(1/2 + Delta/3))."""
        expected = math.degrees(math.asin(math.sqrt(0.5 + DELTA_F / 3.0)))
        assert abs(THETA_23_GEO - expected) < 1e-12

    def test_theta_23_value(self):
        """theta_23 ~ 48.99 deg."""
        assert abs(THETA_23_GEO - 48.99) < 0.01

    def test_theta_23_vs_observed(self):
        """theta_23 within 1 sigma of observed (49.0 +/- 1.4)."""
        obs = 49.0
        unc = 1.4
        sigma = abs(THETA_23_GEO - obs) / unc
        assert sigma < 1.0

    def test_theta_13_formula(self):
        """theta_13 = arcsin(Delta/sqrt2)."""
        expected = math.degrees(math.asin(DELTA_F / math.sqrt(2.0)))
        assert abs(THETA_13_GEO - expected) < 1e-12

    def test_theta_13_value(self):
        """theta_13 ~ 8.47 deg."""
        assert abs(THETA_13_GEO - 8.47) < 0.01

    def test_theta_13_vs_observed(self):
        """theta_13 within 1 sigma of observed (8.54 +/- 0.15)."""
        obs = 8.54
        unc = 0.15
        sigma = abs(THETA_13_GEO - obs) / unc
        assert sigma < 1.0

    def test_all_angles_positive(self):
        """All angles are positive."""
        assert THETA_12_GEO > 0
        assert THETA_23_GEO > 0
        assert THETA_13_GEO > 0

    def test_hierarchy(self):
        """theta_23 > theta_12 > theta_13 (normal hierarchy)."""
        assert THETA_23_GEO > THETA_12_GEO > THETA_13_GEO


class TestParameterDependence:
    """Test sensitivity to input parameters."""

    def test_theta_12_with_zero_delta(self):
        """With delta_g=0, theta_12 = TBM exactly."""
        val = theta_12_geo(delta_g=0.0)
        assert abs(val - THETA_12_TBM) < 1e-12

    def test_theta_23_with_zero_delta(self):
        """With delta=0, theta_23 = 45 exactly."""
        val = theta_23_geo(delta=0.0)
        assert abs(val - 45.0) < 1e-12

    def test_theta_13_with_zero_delta(self):
        """With delta=0, theta_13 = 0 exactly (TBM limit)."""
        val = theta_13_geo(delta=0.0)
        assert abs(val) < 1e-12


class TestOverlapMatrix:
    """Test the 3x3 cone overlap matrix."""

    def test_diagonal_is_unity(self):
        """Diagonal entries are 1.0 (self-overlap)."""
        for i in range(3):
            assert abs(OVERLAP_MATRIX[i][i] - 1.0) < 1e-15

    def test_symmetric(self):
        """Matrix is symmetric."""
        for i in range(3):
            for j in range(3):
                assert abs(OVERLAP_MATRIX[i][j] - OVERLAP_MATRIX[j][i]) < 1e-15

    def test_off_diagonal_positive(self):
        """Off-diagonal entries are small and positive."""
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert 0 < OVERLAP_MATRIX[i][j] < 0.5

    def test_off_diagonal_equal(self):
        """All off-diagonal entries equal (S_3 symmetry)."""
        f01 = OVERLAP_MATRIX[0][1]
        f02 = OVERLAP_MATRIX[0][2]
        f12 = OVERLAP_MATRIX[1][2]
        assert abs(f01 - f02) < 1e-15
        assert abs(f01 - f12) < 1e-15

    def test_size(self):
        """Matrix is 3x3."""
        assert len(OVERLAP_MATRIX) == 3
        for row in OVERLAP_MATRIX:
            assert len(row) == 3

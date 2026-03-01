"""Tests for the inflationary observables."""

from fractions import Fraction

from sdgft.inflation import (
    e_folds, spectral_index, tensor_to_scalar,
    slow_roll_epsilon, slow_roll_eta,
    BETA_ISO, BETA_ISO_F,
    N_EFOLDS_FP, N_EFOLDS_TREE,
    N_S, R_TENSOR,
    EPSILON_SR, ETA_SR,
)
from sdgft.constants import DELTA_F, DELTA_G_F
from sdgft.dimension import D_STAR_TREE_F, D_STAR_FP, N_TREE_F


class TestEfolds:

    def test_n_efolds_fp_value(self):
        """N_e ~ 60 with D*_fp."""
        assert abs(N_EFOLDS_FP - 60.0) < 1.0

    def test_n_efolds_tree_value(self):
        """N_e ~ 60 with D*_tree (slightly less)."""
        assert abs(N_EFOLDS_TREE - 60.0) < 1.0

    def test_n_efolds_fp_greater(self):
        """FP gives slightly more e-folds than tree."""
        assert N_EFOLDS_FP > N_EFOLDS_TREE

    def test_e_folds_function(self):
        result = e_folds(D_STAR_FP, DELTA_F, DELTA_G_F)
        assert abs(result - N_EFOLDS_FP) < 1e-10


class TestSpectralIndex:

    def test_n_s_value(self):
        """n_s ~ 0.967."""
        assert abs(N_S - 0.967) < 0.001

    def test_n_s_within_planck_1sigma(self):
        """n_s should be within 1 sigma of Planck (0.9649 +/- 0.0042)."""
        assert abs(N_S - 0.9649) < 0.0042

    def test_n_s_less_than_one(self):
        """Red tilt: n_s < 1."""
        assert N_S < 1.0

    def test_spectral_index_function(self):
        result = spectral_index(N_TREE_F, N_EFOLDS_FP)
        assert abs(result - N_S) < 1e-12


class TestTensorRatio:

    def test_r_value(self):
        """r ~ 0.013."""
        assert abs(R_TENSOR - 0.013) < 0.001

    def test_r_below_bicep_limit(self):
        """r < 0.036 (BICEP/Keck 95% CL)."""
        assert R_TENSOR < 0.036

    def test_r_positive(self):
        assert R_TENSOR > 0

    def test_tensor_to_scalar_function(self):
        result = tensor_to_scalar(N_TREE_F, N_EFOLDS_FP)
        assert abs(result - R_TENSOR) < 1e-12


class TestIsocurvature:

    def test_beta_iso_fraction(self):
        assert BETA_ISO == Fraction(1, 36)

    def test_beta_iso_value(self):
        assert abs(BETA_ISO_F - 1.0 / 36.0) < 1e-15

    def test_beta_iso_below_planck_limit(self):
        """beta_iso < 0.038 (Planck 95% CL)."""
        assert BETA_ISO_F < 0.038


class TestSlowRoll:

    def test_epsilon_sr_small(self):
        """epsilon << 1 (slow-roll condition)."""
        assert EPSILON_SR < 0.01
        assert EPSILON_SR > 0

    def test_epsilon_sr_value(self):
        """epsilon ~ 5.5e-5 for n = 67/48, N_e ~ 60."""
        assert abs(EPSILON_SR - 5.5e-5) < 1e-5

    def test_eta_sr_small(self):
        """eta << 1 (slow-roll condition)."""
        assert abs(ETA_SR) < 0.1

    def test_eta_sr_negative(self):
        """Second slow-roll parameter is negative for n = 67/48."""
        assert ETA_SR < 0

    def test_eta_sr_value(self):
        """eta ~ -0.017."""
        assert abs(ETA_SR - (-0.017)) < 0.002

    def test_slow_roll_epsilon_function(self):
        result = slow_roll_epsilon(N_TREE_F, N_EFOLDS_FP)
        assert abs(result - EPSILON_SR) < 1e-15

    def test_slow_roll_eta_function(self):
        result = slow_roll_eta(N_TREE_F, N_EFOLDS_FP)
        assert abs(result - ETA_SR) < 1e-15

    def test_consistency_r_not_16_epsilon(self):
        """In f(R) gravity, r != 16*epsilon (modified consistency)."""
        assert abs(R_TENSOR - 16.0 * EPSILON_SR) > 0.01

    def test_r_from_epsilon(self):
        """r = 48(2n-1)/(n-2)^2 * epsilon in f(R)."""
        n = N_TREE_F
        factor = 48.0 * (2.0 * n - 1.0) / (n - 2.0) ** 2
        assert abs(R_TENSOR - factor * EPSILON_SR) < 1e-10

    def test_epsilon_gr_limit(self):
        """For n -> 1, epsilon -> 0 (de Sitter)."""
        eps = slow_roll_epsilon(1.0001, 60.0)
        assert eps < 1e-3  # much smaller than SDGFT value

"""Tests for the particle physics observables.

NOTE: Some values differ from the monograph (ch04_particle_physics.tex):
- theta_12: Replaced the old formula arctan(1/sqrt(2)) - Delta*pi/6 (= 29.0°, 5.48σ)
  with the geometric TBM formula arctan(1/sqrt(2)) * (1 - 1/24) (= 33.80°, 0.51σ).
- theta_23: Formula yields 48.8°, not 47.6°. Actually closer to observed 49.0°.
- m_mu/m_e: Using SDGFT's own alpha_em (~136.9) instead of the observed (137.036)
  gives ~206.56 instead of the monograph's 206.762.

These discrepancies are flagged for review in the Observable registry.
"""

import math

from sdgft.constants import DELTA_F, DELTA_G_F, PHI
from sdgft.dimension import D_STAR_TREE_F, D_STAR_FP
from sdgft.cosmology import OMEGA_B_F
from sdgft.particle import (
    alpha_em_inv, alpha_em, alpha_s_value, sin2_theta_w,
    mu_e_ratio, tau_mu_ratio, lambda_geo, higgs_mass,
    n_generations, theta_12, theta_23, theta_13,
    v_us, v_ub, quark_hierarchy,
    ALPHA_EM_INV_TREE, ALPHA_EM_INV_FP,
    ALPHA_EM_TREE, ALPHA_S, SIN2_THETA_W,
    MU_E_RATIO, TAU_MU_RATIO_TREE, TAU_MU_RATIO_FP,
    TAU_E_RATIO_TREE,
    LAMBDA_GEO, HIGGS_MASS, N_GEN,
    THETA_12, THETA_23, THETA_13,
    V_US, V_UB, QUARK_HIERARCHY,
    GAMMA_EW, V_HIGGS,
)


class TestCouplingConstants:

    def test_alpha_em_inv_tree(self):
        """alpha_em^{-1} ~ 137 (tree)."""
        assert abs(ALPHA_EM_INV_TREE - 137.0) < 1.0

    def test_alpha_em_inv_fp(self):
        """alpha_em^{-1} ~ 137.5 (fp)."""
        assert abs(ALPHA_EM_INV_FP - 137.5) < 0.5

    def test_alpha_em_inv_function(self):
        result = alpha_em_inv(D_STAR_TREE_F, DELTA_G_F)
        assert abs(result - ALPHA_EM_INV_TREE) < 1e-12

    def test_alpha_em_inverse(self):
        assert abs(ALPHA_EM_TREE * ALPHA_EM_INV_TREE - 1.0) < 1e-14

    def test_alpha_s_value(self):
        """alpha_s ~ 0.1179."""
        assert abs(ALPHA_S - math.sqrt(2) / 12.0) < 1e-15

    def test_alpha_s_close_to_observed(self):
        """alpha_s = 0.11785 vs observed 0.1179 +/- 0.0009."""
        assert abs(ALPHA_S - 0.1179) < 0.0009

    def test_sin2_theta_w(self):
        """sin^2(theta_W) ~ 0.2311."""
        assert abs(SIN2_THETA_W - 0.2311) < 0.001

    def test_sin2_theta_w_close_to_observed(self):
        """0.23122 +/- 0.00003."""
        assert abs(SIN2_THETA_W - 0.23122) < 0.001

    def test_sin2_theta_w_tree_level(self):
        """Tree level is 1/9."""
        assert abs(sin2_theta_w(0.0) - 1.0 / 9.0) < 1e-15


class TestLeptonMasses:

    def test_mu_e_ratio_value(self):
        """m_mu/m_e ~ 206.5 (using SDGFT's own alpha_em)."""
        assert abs(MU_E_RATIO - 206.5) < 1.0

    def test_mu_e_ratio_close_to_observed(self):
        """Observed: 206.768. Using SDGFT alpha_em gives ~0.2% deviation."""
        pct = abs(MU_E_RATIO - 206.768) / 206.768 * 100
        assert pct < 0.5

    def test_mu_e_ratio_with_observed_alpha(self):
        """With observed alpha_em, should match monograph's 206.762."""
        result = mu_e_ratio(1.0 / 137.036, DELTA_F)
        assert abs(result - 206.762) < 0.1

    def test_tau_mu_ratio_tree(self):
        """m_tau/m_mu = 6*D*_tree = 6*67/24 = 402/24 = 16.75."""
        assert abs(TAU_MU_RATIO_TREE - 16.75) < 0.01

    def test_tau_mu_ratio_close_to_observed(self):
        """Observed: 16.817."""
        pct = abs(TAU_MU_RATIO_TREE - 16.817) / 16.817 * 100
        assert pct < 1.0

    def test_tau_e_ratio_composite(self):
        """m_tau/m_e = (m_mu/m_e) * (m_tau/m_mu)."""
        expected = MU_E_RATIO * TAU_MU_RATIO_TREE
        assert abs(TAU_E_RATIO_TREE - expected) < 1e-10


class TestHiggsSector:

    def test_lambda_geo_value(self):
        """lambda = Delta/phi ~ 0.1288."""
        expected = DELTA_F / PHI
        assert abs(LAMBDA_GEO - expected) < 1e-12

    def test_higgs_mass_value(self):
        """m_H ~ 124.94 GeV."""
        assert abs(HIGGS_MASS - 124.94) < 0.5

    def test_higgs_mass_close_to_observed(self):
        """Observed: 125.25 +/- 0.17 GeV."""
        assert abs(HIGGS_MASS - 125.25) < 0.5

    def test_higgs_mass_formula(self):
        result = higgs_mass(DELTA_F, PHI, V_HIGGS)
        assert abs(result - HIGGS_MASS) < 1e-10


class TestGenerations:

    def test_n_gen_is_3(self):
        assert N_GEN == 3

    def test_phi3_less_than_5(self):
        assert PHI ** 3 < 5.0

    def test_phi4_greater_than_5(self):
        assert PHI ** 4 > 5.0

    def test_n_generations_function(self):
        assert n_generations(PHI) == 3


class TestNeutrinoMixing:
    """Neutrino mixing angle tests.

    theta_12 uses the geometric TBM formula: arctan(1/sqrt(2)) * (1 - delta_g)
    = arctan(1/sqrt(2)) * 23/24 = 33.80 deg (0.51 sigma from observed 33.41).
    """

    def test_theta_12_formula_correct(self):
        """arctan(1/sqrt(2)) * (1 - 1/24) in degrees."""
        tbm = math.atan(1 / math.sqrt(2))
        expected = math.degrees(tbm * (1.0 - DELTA_G_F))
        assert abs(THETA_12 - expected) < 1e-10

    def test_theta_12_value(self):
        """TBM * 23/24 ~ 33.80 deg."""
        assert abs(THETA_12 - 33.80) < 0.01

    def test_theta_12_close_to_observed(self):
        """Observed: 33.41 +/- 0.75 deg. Within 1 sigma."""
        sigma = abs(THETA_12 - 33.41) / 0.75
        assert sigma < 1.0

    def test_theta_12_tbm_limit(self):
        """With delta_g=0, theta_12 = TBM ~ 35.26 deg."""
        val = theta_12(0.0)
        expected = math.degrees(math.atan(1 / math.sqrt(2)))
        assert abs(val - expected) < 1e-12

    def test_theta_23_formula_correct(self):
        """45*(1 + Delta/sqrt(6)) in degrees."""
        expected = 45.0 * (1.0 + DELTA_F / math.sqrt(6.0))
        assert abs(THETA_23 - expected) < 1e-10

    def test_theta_23_actual_value(self):
        """Formula yields ~48.8°, closer to observed 49.0° than claimed 47.6°."""
        assert abs(THETA_23 - 48.8) < 0.5

    def test_theta_13_value(self):
        """theta_13 ~ 8.47 deg."""
        assert abs(THETA_13 - 8.47) < 0.1

    def test_theta_13_close_to_observed(self):
        """Observed: 8.54 +/- 0.15 deg."""
        assert abs(THETA_13 - 8.54) < 0.15


class TestCKM:

    def test_v_us_value(self):
        """|V_us| ~ 0.2234."""
        assert abs(V_US - 0.2234) < 0.001

    def test_v_us_close_to_observed(self):
        """Observed: 0.2243 +/- 0.0008."""
        assert abs(V_US - 0.2243) < 0.002

    def test_v_us_is_sqrt_omega_b(self):
        assert abs(V_US - math.sqrt(OMEGA_B_F)) < 1e-12

    def test_v_ub_value(self):
        """|V_ub| ~ 0.00375."""
        assert abs(V_UB - 0.00375) < 0.001

    def test_v_ub_close_to_observed(self):
        """Observed: 0.00382 +/- 0.0002."""
        assert abs(V_UB - 0.00382) < 0.001


class TestExternalConstants:

    def test_gamma_ew_documented(self):
        assert abs(GAMMA_EW - 0.12011) < 1e-5

    def test_v_higgs_documented(self):
        assert abs(V_HIGGS - 246.22) < 0.01


class TestQuarkHierarchy:

    def test_confinement_value(self):
        """exp(2*pi) ~ 535.5 at alpha_s = 1."""
        expected = math.exp(2 * math.pi)
        assert abs(QUARK_HIERARCHY - expected) < 0.1

    def test_close_to_mc_over_mu(self):
        """m_c/m_u ~ 590; deviation < 15%."""
        mc_over_mu = 1275.0 / 2.16  # ~ 590
        deviation = abs(QUARK_HIERARCHY - mc_over_mu) / mc_over_mu
        assert deviation < 0.15

    def test_function_alpha_s_1(self):
        """quark_hierarchy(1.0) == exp(2*pi)."""
        assert abs(quark_hierarchy(1.0) - math.exp(2 * math.pi)) < 1e-10

    def test_function_monotone_decreasing(self):
        """Larger alpha_s => smaller ratio."""
        assert quark_hierarchy(0.5) > quark_hierarchy(1.0)

    def test_perturbative_scale_huge(self):
        """At M_Z, exp(2*pi/0.118) ~ 10^23 (too large to be physical)."""
        val = quark_hierarchy(ALPHA_S)
        assert val > 1e20

    def test_constant_uses_confinement(self):
        """QUARK_HIERARCHY should be exp(2*pi), not exp(2*pi/0.118)."""
        assert QUARK_HIERARCHY < 1000  # sanity: ~535, not ~10^23

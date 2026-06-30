"""Tests for the RG running module."""

import math
import pytest

from sdgft.experimental.rg_running import (
    M_Z, M_PL, T_PL, B1, B2, B3,
    ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ,
    SIN2_THETA_W_PLANCK,
    couplings_from_observables,
    sin2_from_inv_couplings,
    alpha_em_inv_from_couplings,
    run_inverse_couplings,
    run_to_scale,
    find_unification_scale,
    rg_trajectory,
    SIN2_THETA_W_SM_PLANCK,
    GAMMA_EW_ARITHMETIC,
    SIN2_THETA_W_RG,
    ALPHA_RATIO_SDGFT,
    T_GUT, M_GUT,
    ALPHA_S_MZ_EXACT,
    run_2loop_rge,
    MZ,
    MT,
)



class TestCouplingConversion:
    """Test conversion between observables and inverse couplings."""

    def test_roundtrip(self):
        """Convert to couplings and back, verify consistency."""
        ia1, ia2, ia3 = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        sw = sin2_from_inv_couplings(ia1, ia2)
        aem_inv = alpha_em_inv_from_couplings(ia1, ia2)
        assert abs(sw - SIN2_THETA_W_MZ) < 1e-12
        assert abs(aem_inv - ALPHA_EM_INV_MZ) < 1e-10

    def test_inv_alpha_1_value(self):
        ia1, _, _ = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        assert abs(ia1 - 59.02) < 0.1

    def test_inv_alpha_2_value(self):
        _, ia2, _ = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        assert abs(ia2 - 29.58) < 0.1

    def test_inv_alpha_3_value(self):
        _, _, ia3 = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        assert abs(ia3 - 1.0 / 0.1179) < 0.01


class TestAnalyticRunning:
    """Test the analytic 1-loop running."""

    def test_no_running_at_mz(self):
        """At delta_t = 0, couplings unchanged."""
        ia1, ia2, ia3 = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        r1, r2, r3 = run_inverse_couplings(ia1, ia2, ia3, 0.0)
        assert abs(r1 - ia1) < 1e-14
        assert abs(r2 - ia2) < 1e-14
        assert abs(r3 - ia3) < 1e-14

    def test_alpha_1_increases_with_energy(self):
        """U(1)_Y coupling increases (1/alpha_1 decreases) with energy."""
        ia1, ia2, ia3 = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        r1, _, _ = run_inverse_couplings(ia1, ia2, ia3, T_PL)
        assert r1 < ia1  # 1/alpha_1 decreases => alpha_1 increases

    def test_alpha_2_decreases_with_energy(self):
        """SU(2)_L coupling decreases (1/alpha_2 increases) with energy."""
        ia1, ia2, ia3 = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        _, r2, _ = run_inverse_couplings(ia1, ia2, ia3, T_PL)
        assert r2 > ia2  # 1/alpha_2 increases => alpha_2 decreases

    def test_alpha_3_decreases_with_energy(self):
        """SU(3)_c is asymptotically free (1/alpha_3 increases)."""
        ia1, ia2, ia3 = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        _, _, r3 = run_inverse_couplings(ia1, ia2, ia3, T_PL)
        assert r3 > ia3


class TestSMRunning:
    """Test SM running results."""

    def test_sin2_theta_w_at_planck_sm(self):
        """SM running gives sin^2(theta_W) ~ 0.47 at M_Pl, NOT 1/9."""
        assert SIN2_THETA_W_SM_PLANCK > 0.4
        assert SIN2_THETA_W_SM_PLANCK < 0.5
        # Definitely not 1/9
        assert abs(SIN2_THETA_W_SM_PLANCK - 1.0 / 9.0) > 0.3

    def test_gamma_ew_arithmetic(self):
        """gamma_EW = sin^2(theta_W)(M_Z) - 1/9 ~ 0.120."""
        assert abs(GAMMA_EW_ARITHMETIC - 0.120) < 0.001

    def test_sin2_theta_w_rg_equals_observed(self):
        """1/9 + gamma_EW trivially equals the observed value."""
        assert abs(SIN2_THETA_W_RG - SIN2_THETA_W_MZ) < 1e-14


class TestGUTScale:
    """Test GUT unification scale."""

    def test_gut_scale_order(self):
        """GUT scale should be between 10^12 and 10^16 GeV."""
        assert 1e12 < M_GUT < 1e16

    def test_unification_at_gut(self):
        """At T_GUT, alpha_1 = alpha_2."""
        result = run_to_scale(T_GUT)
        assert abs(result["inv_alpha_1"] - result["inv_alpha_2"]) < 0.01


class TestSDGFTConnection:
    """Test SDGFT-specific results."""

    def test_alpha_ratio_is_delta(self):
        """If sin^2(theta_W)(M_Pl) = 1/9, then alpha_1/alpha_2 = 5/24 = Delta."""
        assert abs(ALPHA_RATIO_SDGFT - 5.0 / 24.0) < 1e-14

    def test_t_pl_value(self):
        """t_Pl = ln(M_Pl/M_Z) ~ 39.47."""
        assert abs(T_PL - 39.47) < 0.1


class TestTrajectory:
    """Test the trajectory function."""

    def test_trajectory_starts_at_mz(self):
        traj = rg_trajectory(10)
        assert abs(traj[0]["sin2_theta_w"] - SIN2_THETA_W_MZ) < 1e-10

    def test_trajectory_ends_at_planck(self):
        traj = rg_trajectory(10)
        assert abs(traj[-1]["sin2_theta_w"] - SIN2_THETA_W_SM_PLANCK) < 0.01

    def test_trajectory_monotonic_sin2(self):
        """sin^2(theta_W) increases with energy in the SM."""
        traj = rg_trajectory(50)
        for i in range(1, len(traj)):
            assert traj[i]["sin2_theta_w"] >= traj[i - 1]["sin2_theta_w"] - 1e-10


class TestTwoLoopRGE:
    """Test the 2-loop RGE solver with top quark mass threshold."""

    def test_exact_alpha_s_initial_value(self):
        """Verifies that the exact SDGFT value is mathematically sound and matches expectations."""
        assert ALPHA_S_MZ_EXACT == pytest.approx(0.11785113, abs=1e-6)

    def test_2loop_rge_running_to_top_threshold(self):
        """Tests the 2-loop RGE solver integration from M_Z up to the Top-Threshold scale."""
        # Standard values at M_Z: inverse alpha for U(1)_Y (GUT normalized), SU(2)_L, and SU(3)_c
        # alpha_1_inv ~ 58.98, alpha_2_inv ~ 29.57, alpha_3_inv = 1 / ALPHA_S_MZ_EXACT
        alpha_inv_mz = [58.98, 29.57, 1.0 / ALPHA_S_MZ_EXACT]
        
        alpha_inv_mt = run_2loop_rge(MT, alpha_inv_mz, steps=200)
        
        # Strong coupling should decrease (inverse alpha_3 increases) as energy scale goes up
        assert alpha_inv_mt[2] > alpha_inv_mz[2]
        # Check that alpha_s at M_t is running correctly towards smaller values (~0.108)
        alpha_s_mt = 1.0 / alpha_inv_mt[2]
        assert alpha_s_mt == pytest.approx(0.108, abs=0.005)

    def test_2loop_rge_running_to_planck_scale(self):
        """Verifies convergence trends up to the high energy scale."""
        alpha_inv_mz = [58.98, 29.57, 1.0 / ALPHA_S_MZ_EXACT]
        m_pl = 1.22e19  # Planck scale in GeV
        
        alpha_inv_mpl = run_2loop_rge(m_pl, alpha_inv_mz, steps=2000)
        
        # Asymptotics: alpha_1_inv should decrease (coupling gets stronger)
        assert alpha_inv_mpl[0] < alpha_inv_mz[0]
        # alpha_3_inv should significantly increase (strong coupling gets weaker due to asymptotic freedom)
        assert alpha_inv_mpl[2] > 40.0


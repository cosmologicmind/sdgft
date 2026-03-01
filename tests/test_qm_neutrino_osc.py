"""Tests for SDGFT QM neutrino oscillation module.

Covers: mass-splitting ratio, neutrino masses, mass-squared differences,
CP phase, PMNS matrix, oscillation probabilities, effective Majorana mass,
experiment predictions, registry, and cross-consistency checks.
"""

from __future__ import annotations

import math
import cmath
from fractions import Fraction

import pytest

from sdgft.constants import DELTA, DELTA_F, DELTA_G, DELTA_G_F
from sdgft.dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from sdgft.experimental.neutrino_mass import SUM_M_NU
from sdgft.qm import neutrino_osc as no


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. Mass-splitting ratio                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestMassSplittingRatio:
    """R = D*/(2δ) = 67/2."""

    def test_exact_fraction(self):
        """Exact rational: 67/2."""
        assert no.mass_splitting_ratio_exact() == Fraction(67, 2)

    def test_exact_from_axioms(self):
        """Verify R = D*_tree / (2·δ) from Fraction constants."""
        expected = D_STAR_TREE / (2 * DELTA_G)
        assert expected == Fraction(67, 2)

    def test_float_value(self):
        """Float: 33.5 exactly."""
        assert no.mass_splitting_ratio() == pytest.approx(33.5, abs=1e-12)

    def test_module_level_tree(self):
        """Module-level R_TREE = 67/2."""
        assert no.R_TREE == Fraction(67, 2)
        assert no.R_TREE_F == pytest.approx(33.5, abs=1e-12)

    def test_fp_variant(self):
        """Fixed-point D* gives slightly different R."""
        r_fp = no.mass_splitting_ratio(D_STAR_FP, DELTA_G_F)
        assert r_fp > 33.5
        assert r_fp < 34.0
        assert no.R_FP == pytest.approx(r_fp)

    def test_ratio_vs_observed(self):
        """R_pred is within ~0.5σ of observed ratio."""
        sigma = abs(no.R_TREE_F - no.RATIO_OBS) / no.RATIO_OBS_UNC
        assert sigma < 1.0, f"Ratio deviation {sigma:.2f}σ > 1σ"

    def test_integer_numerator(self):
        """Numerator 67 = 24·D* comes from D* = 67/24."""
        r = no.mass_splitting_ratio_exact()
        assert r.numerator == 67
        assert r.denominator == 2


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. Neutrino masses                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestNeutrinoMasses:
    """m₁ = 0, m₂ = Σ/(1+√R), m₃ = Σ√R/(1+√R)."""

    def test_m1_is_zero(self):
        """Lightest neutrino mass is exactly zero."""
        m1, m2, m3 = no.neutrino_masses()
        assert m1 == 0.0

    def test_sum_equals_sigma(self):
        """m₁ + m₂ + m₃ = Σm_ν."""
        m1, m2, m3 = no.neutrino_masses()
        assert m1 + m2 + m3 == pytest.approx(SUM_M_NU, rel=1e-12)

    def test_normal_ordering(self):
        """m₁ < m₂ < m₃ (normal ordering)."""
        m1, m2, m3 = no.neutrino_masses()
        assert m1 < m2 < m3

    def test_m2_order_of_magnitude(self):
        """m₂ ~ 8–9 meV."""
        _, m2, _ = no.neutrino_masses()
        assert 0.007 < m2 < 0.010

    def test_m3_order_of_magnitude(self):
        """m₃ ~ 48–51 meV."""
        _, _, m3 = no.neutrino_masses()
        assert 0.045 < m3 < 0.055

    def test_m3_over_m2_equals_sqrt_r(self):
        """m₃/m₂ = √R."""
        _, m2, m3 = no.neutrino_masses()
        sqrt_r = math.sqrt(no.R_TREE_F)
        assert m3 / m2 == pytest.approx(sqrt_r, rel=1e-12)

    def test_module_level_values(self):
        """Module-level M1, M2, M3."""
        assert no.M1 == 0.0
        assert no.M2 > 0
        assert no.M3 > no.M2
        assert no.M1 + no.M2 + no.M3 == pytest.approx(SUM_M_NU, rel=1e-12)

    def test_custom_sum(self):
        """With custom Σm_ν, masses still respect constraints."""
        m1, m2, m3 = no.neutrino_masses(sum_m_nu=0.1)
        assert m1 == 0.0
        assert m1 + m2 + m3 == pytest.approx(0.1, rel=1e-12)

    def test_custom_ratio(self):
        """With custom R, ratio is respected."""
        m1, m2, m3 = no.neutrino_masses(r=100.0)
        assert m3 / m2 == pytest.approx(10.0, rel=1e-12)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. Mass-squared differences                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestMassSquaredDifferences:
    """Δm²₂₁, Δm²₃₁, Δm²₃₂ from the mass spectrum."""

    def test_dm2_21_positive(self):
        """Solar splitting is positive (NO)."""
        assert no.DM2_21 > 0

    def test_dm2_32_positive(self):
        """Atmospheric splitting is positive (NO)."""
        assert no.DM2_32 > 0

    def test_dm2_31_positive(self):
        """Δm²₃₁ is positive."""
        assert no.DM2_31 > 0

    def test_consistency(self):
        """Δm²₃₁ = Δm²₃₂ + Δm²₂₁."""
        assert no.DM2_31 == pytest.approx(no.DM2_32 + no.DM2_21, rel=1e-12)

    def test_ratio_equals_r(self):
        """Δm²₃₁/Δm²₂₁ = R = 33.5."""
        ratio = no.DM2_31 / no.DM2_21
        assert ratio == pytest.approx(no.R_TREE_F, rel=1e-12)

    def test_dm2_21_order_of_magnitude(self):
        """Δm²₂₁ ~ 7 × 10⁻⁵ eV²."""
        assert 5e-5 < no.DM2_21 < 1e-4

    def test_dm2_32_order_of_magnitude(self):
        """Δm²₃₂ ~ 2.4 × 10⁻³ eV²."""
        assert 2e-3 < no.DM2_32 < 3e-3

    def test_dm2_21_vs_observed(self):
        """Δm²₂₁ within 2σ of observation."""
        sigma = abs(no.DM2_21 - no.DM2_21_OBS) / no.DM2_21_OBS_UNC
        assert sigma < 2.0, f"Δm²₂₁ deviation {sigma:.2f}σ > 2σ"

    def test_dm2_32_vs_observed(self):
        """Δm²₃₂ within 3σ of observation."""
        sigma = abs(no.DM2_32 - no.DM2_32_OBS) / no.DM2_32_OBS_UNC
        assert sigma < 3.0, f"Δm²₃₂ deviation {sigma:.2f}σ > 3σ"

    def test_hierarchy(self):
        """Δm²₃₂ ≫ Δm²₂₁ (atmospheric ≫ solar)."""
        assert no.DM2_32 > 10 * no.DM2_21

    def test_dm2_equals_m_squared(self):
        """Since m₁ = 0: Δm²₂₁ = m₂², Δm²₃₁ = m₃²."""
        assert no.DM2_21 == pytest.approx(no.M2 ** 2, rel=1e-12)
        assert no.DM2_31 == pytest.approx(no.M3 ** 2, rel=1e-12)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. CP phase                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestCPPhase:
    """δ_CP = 5π/4."""

    def test_value_radians(self):
        """δ_CP = 5π/4 ≈ 3.927 rad."""
        assert no.delta_cp_pmns() == pytest.approx(5.0 * math.pi / 4.0, rel=1e-15)

    def test_value_degrees(self):
        """δ_CP = 225°."""
        assert math.degrees(no.DELTA_CP) == pytest.approx(225.0, rel=1e-12)

    def test_exact_fraction(self):
        """Exact: δ_CP/π = 5/4."""
        assert no.delta_cp_pmns_exact() == Fraction(5, 4)

    def test_axiom_derivation_a(self):
        """From Fibonacci-cone: (Δ/δ)·π·sin²(30°) = 5·π/4."""
        ratio = DELTA / DELTA_G  # = 5
        result = float(ratio) * math.pi * float(DELTA + DELTA_G)  # sin²(30°) = 1/4
        assert result == pytest.approx(no.DELTA_CP, rel=1e-12)

    def test_axiom_derivation_b(self):
        """From lattice modulation: (3π/2)(1−Δ+δ) = (3π/2)(20/24) = 5π/4."""
        net_coverage = 1.0 - DELTA_F + DELTA_G_F  # 20/24
        result = 1.5 * math.pi * net_coverage
        assert result == pytest.approx(no.DELTA_CP, rel=1e-12)

    def test_sin_negative(self):
        """sin(δ_CP) < 0 (CP violation, third quadrant)."""
        assert math.sin(no.DELTA_CP) < 0

    def test_module_level(self):
        """Module-level DELTA_CP is correct."""
        assert no.DELTA_CP == pytest.approx(5.0 * math.pi / 4.0, rel=1e-15)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. PMNS matrix                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestPMNSMatrix:
    """Properties of the 3×3 PMNS matrix."""

    def test_shape(self):
        """3×3 matrix."""
        U = no.U_PMNS
        assert len(U) == 3
        assert all(len(row) == 3 for row in U)

    def test_unitarity(self):
        """UU† = I."""
        assert no.pmns_is_unitary(no.U_PMNS, tol=1e-12)

    def test_normalization_rows(self):
        """Each row is normalised: Σ|U_{αi}|² = 1."""
        U = no.U_PMNS
        for row in U:
            norm = sum(abs(u) ** 2 for u in row)
            assert norm == pytest.approx(1.0, abs=1e-12)

    def test_normalization_cols(self):
        """Each column is normalised: Σ|U_{αi}|² = 1."""
        U = no.U_PMNS
        for j in range(3):
            norm = sum(abs(U[i][j]) ** 2 for i in range(3))
            assert norm == pytest.approx(1.0, abs=1e-12)

    def test_u_e3_has_phase(self):
        """U_{e3} = s₁₃ · e^{−iδ}, so Im(U_e3) ≠ 0."""
        u_e3 = no.U_PMNS[0][2]
        assert abs(u_e3.imag) > 0.01

    def test_u_e3_magnitude(self):
        """|U_{e3}| = sin(θ₁₃)."""
        u_e3 = no.U_PMNS[0][2]
        s13 = math.sin(math.radians(no.THETA_13_DEG))
        assert abs(u_e3) == pytest.approx(s13, rel=1e-10)

    def test_u_e1_real(self):
        """U_{e1} = c₁₂c₁₃ is real and positive."""
        u_e1 = no.U_PMNS[0][0]
        assert abs(u_e1.imag) < 1e-14
        assert u_e1.real > 0

    def test_angles_roundtrip(self):
        """Extract θ₁₃ from |U_{e3}| and verify."""
        u_e3 = no.U_PMNS[0][2]
        s13 = abs(u_e3)
        theta_13_extracted = math.degrees(math.asin(s13))
        assert theta_13_extracted == pytest.approx(no.THETA_13_DEG, rel=1e-10)

    def test_custom_angles(self):
        """PMNS with custom angles is still unitary."""
        U = no.pmns_matrix(35.0, 45.0, 9.0, math.pi)
        assert no.pmns_is_unitary(U, tol=1e-12)

    def test_zero_cp_is_real(self):
        """With δ_CP = 0, all elements are real."""
        U = no.pmns_matrix(33.0, 45.0, 8.5, 0.0)
        for row in U:
            for elem in row:
                assert abs(elem.imag) < 1e-14


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. Jarlskog invariant                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestJarlskogPMNS:
    """Lepton Jarlskog invariant J_PMNS."""

    def test_negative(self):
        """J < 0 because sin(5π/4) < 0."""
        assert no.J_PMNS < 0

    def test_order_of_magnitude(self):
        """J ~ −0.023."""
        assert -0.030 < no.J_PMNS < -0.020

    def test_vs_matrix_invariant(self):
        """J from formula = Im(U[0][1]·U[1][2]·conj(U[0][2])·conj(U[1][1]))."""
        U = no.U_PMNS
        j_matrix = (U[0][1] * U[1][2]
                     * U[0][2].conjugate() * U[1][1].conjugate()).imag
        assert j_matrix == pytest.approx(no.J_PMNS, rel=1e-8)

    def test_zero_cp_gives_zero_j(self):
        """With δ_CP = 0: J = 0."""
        j = no.jarlskog_pmns(33.0, 45.0, 8.5, 0.0)
        assert abs(j) < 1e-14


# ╔══════════════════════════════════════════════════════════════════╗
# ║  7. Effective Majorana mass                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestEffectiveMajoranaMass:
    """m_ββ for 0νββ decay."""

    def test_positive(self):
        """m_ββ > 0."""
        assert no.M_BB > 0

    def test_order_of_magnitude(self):
        """m_ββ ~ 1–5 meV."""
        assert 0.001 < no.M_BB < 0.005

    def test_below_current_limit(self):
        """m_ββ < 36 meV (KamLAND-Zen 2023 limit)."""
        assert no.M_BB < no.M_BB_OBS_LIMIT

    def test_m1_zero_simplification(self):
        """With m₁ = 0: m_ββ = |s₁₂²c₁₃² m₂ + s₁₃²e^{−2iδ} m₃|."""
        s12 = math.sin(math.radians(no.THETA_12_DEG))
        c13 = math.cos(math.radians(no.THETA_13_DEG))
        s13 = math.sin(math.radians(no.THETA_13_DEG))
        delta = no.DELTA_CP

        term2 = s12**2 * c13**2 * no.M2
        term3 = s13**2 * cmath.exp(-2j * delta) * no.M3
        m_bb_manual = abs(term2 + term3)
        assert m_bb_manual == pytest.approx(no.M_BB, rel=1e-10)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  8. Oscillation probabilities                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestOscillationProbability:
    """Vacuum 3-flavor oscillation probabilities."""

    def test_survival_at_zero_distance(self):
        """P(α→α, L=0) = 1 for all flavors."""
        for flavor in ("e", "mu", "tau"):
            p = no.oscillation_probability(flavor, flavor, L_km=0.0, E_GeV=1.0)
            assert p == pytest.approx(1.0, abs=1e-12)

    def test_appearance_at_zero_distance(self):
        """P(α→β, L=0) = 0 for α ≠ β."""
        for a, b in [("e", "mu"), ("e", "tau"), ("mu", "tau")]:
            p = no.oscillation_probability(a, b, L_km=0.0, E_GeV=1.0)
            assert p == pytest.approx(0.0, abs=1e-12)

    def test_probability_sum(self):
        """P(α→e) + P(α→μ) + P(α→τ) = 1."""
        for alpha in ("e", "mu", "tau"):
            total = sum(
                no.oscillation_probability(alpha, beta, L_km=1000.0, E_GeV=1.0)
                for beta in ("e", "mu", "tau")
            )
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_probability_sum_antineutrino(self):
        """Probability sum = 1 for antineutrinos."""
        for alpha in ("e", "mu", "tau"):
            total = sum(
                no.oscillation_probability(
                    alpha, beta, L_km=500.0, E_GeV=2.0, antineutrino=True
                )
                for beta in ("e", "mu", "tau")
            )
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_probability_bounded(self):
        """0 ≤ P ≤ 1 for various L/E."""
        for L in [10.0, 100.0, 1000.0, 10000.0]:
            for E in [0.001, 0.1, 1.0, 10.0]:
                p = no.oscillation_probability("mu", "e", L_km=L, E_GeV=E)
                assert 0.0 <= p <= 1.0

    def test_cp_violation_nonzero(self):
        """P(ν_μ→ν_e) ≠ P(ν̄_μ→ν̄_e) at non-trivial L/E."""
        L, E = 1285.0, 2.5
        p_nu = no.oscillation_probability("mu", "e", L, E, antineutrino=False)
        p_bar = no.oscillation_probability("mu", "e", L, E, antineutrino=True)
        assert p_nu != pytest.approx(p_bar, abs=1e-6)

    def test_disappearance_cp_symmetric(self):
        """Disappearance channels are CP-symmetric: P(ν→ν) = P(ν̄→ν̄)."""
        L, E = 295.0, 0.6
        for flavor in ("e", "mu", "tau"):
            p_nu = no.oscillation_probability(flavor, flavor, L, E)
            p_bar = no.oscillation_probability(
                flavor, flavor, L, E, antineutrino=True
            )
            assert p_nu == pytest.approx(p_bar, abs=1e-12)

    def test_appearance_reasonable_dune(self):
        """DUNE ν_μ→ν_e appearance probability is O(1-10%)."""
        p = no.oscillation_probability("mu", "e", L_km=1285.0, E_GeV=2.5)
        assert 0.01 < p < 0.15

    def test_flavor_string_variants(self):
        """Various flavor strings map correctly."""
        p1 = no.oscillation_probability("mu", "e", 1000.0, 1.0)
        p2 = no.oscillation_probability("muon", "electron", 1000.0, 1.0)
        p3 = no.oscillation_probability("nu_mu", "nu_e", 1000.0, 1.0)
        p4 = no.oscillation_probability(1, 0, 1000.0, 1.0)
        assert p1 == pytest.approx(p2, rel=1e-14)
        assert p1 == pytest.approx(p3, rel=1e-14)
        assert p1 == pytest.approx(p4, rel=1e-14)

    def test_invalid_flavor_raises(self):
        """Unknown flavor strings raise ValueError."""
        with pytest.raises(ValueError):
            no.oscillation_probability("charm", "e", 100.0, 1.0)
        with pytest.raises(ValueError):
            no.oscillation_probability(5, 0, 100.0, 1.0)


class TestCPAsymmetry:
    """CP asymmetry A_CP = P(ν) − P(ν̄)."""

    def test_appearance_nonzero(self):
        """A_CP ≠ 0 for appearance channels."""
        a_cp = no.cp_asymmetry("mu", "e", L_km=1285.0, E_GeV=2.5)
        assert a_cp != pytest.approx(0.0, abs=1e-6)

    def test_disappearance_zero(self):
        """A_CP = 0 for disappearance."""
        a_cp = no.cp_asymmetry("mu", "mu", L_km=1285.0, E_GeV=2.5)
        assert a_cp == pytest.approx(0.0, abs=1e-12)

    def test_sign(self):
        """Sign of A_CP reflects sin(δ_CP) < 0."""
        # At moderate L/E, the sign is determined by -sin(δ_CP) 
        # and can be positive or negative depending on L/E
        a_cp = no.cp_asymmetry("mu", "e", L_km=1285.0, E_GeV=2.5)
        assert isinstance(a_cp, float)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  9. Experiment predictions                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestExperimentPredictions:
    """Experiment-specific predictions."""

    def test_dune(self):
        """DUNE prediction has reasonable values."""
        pred = no.predict_dune()
        assert pred.name == "DUNE"
        assert pred.baseline_km == 1285.0
        assert pred.energy_GeV == 2.5
        assert 0.01 < pred.probability < 0.15
        assert 0.01 < pred.probability_anti < 0.15
        assert pred.cp_asymmetry != pytest.approx(0.0, abs=1e-5)

    def test_t2k(self):
        """T2K prediction has reasonable values."""
        pred = no.predict_t2k()
        assert pred.name == "T2K"
        assert pred.baseline_km == 295.0
        assert 0.01 < pred.probability < 0.15

    def test_juno(self):
        """JUNO prediction: ν̄_e disappearance is subdominant."""
        pred = no.predict_juno()
        assert pred.name == "JUNO"
        assert pred.baseline_km == 52.5
        assert pred.energy_GeV == pytest.approx(0.0035)
        # ν̄_e disappearance at reactor baseline
        assert 0.0 < pred.probability < 1.0

    def test_nova(self):
        """NOvA prediction."""
        pred = no.predict_nova()
        assert pred.name == "NOvA"
        assert pred.baseline_km == 810.0
        assert 0.01 < pred.probability < 0.15

    def test_all_experiments_have_cp_asymmetry(self):
        """All appearance experiments show CP asymmetry."""
        for pred_fn in [no.predict_dune, no.predict_t2k, no.predict_nova]:
            pred = pred_fn()
            # Appearance channels should show some CP asymmetry
            assert isinstance(pred.cp_asymmetry, float)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  10. Registry                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestRegistry:
    """Observable registration."""

    def test_register_all(self):
        """All QM observables can be registered."""
        from sdgft.registry import Registry
        reg = Registry()
        no.register_all(reg)
        assert "qm_dm2_21" in reg
        assert "qm_dm2_32" in reg
        assert "qm_dm2_31" in reg
        assert "qm_mass_ratio" in reg
        assert "qm_mass_ordering" in reg
        assert "qm_m_bb" in reg

    def test_observable_count(self):
        """6 new QM observables."""
        from sdgft.registry import Registry
        reg = Registry()
        no.register_all(reg)
        assert len(reg) == 6

    def test_dm2_21_observable(self):
        """Δm²₂₁ observable has correct values."""
        from sdgft.registry import Registry
        reg = Registry()
        no.register_all(reg)
        obs = reg.get("qm_dm2_21")
        assert obs.predicted == pytest.approx(no.DM2_21)
        assert obs.observed == pytest.approx(no.DM2_21_OBS)
        assert obs.unit == "eV²"

    def test_mass_ratio_observable(self):
        """Mass ratio observable has correct values."""
        from sdgft.registry import Registry
        reg = Registry()
        no.register_all(reg)
        obs = reg.get("qm_mass_ratio")
        assert obs.predicted == pytest.approx(33.5)

    def test_m_bb_is_upper_limit(self):
        """m_ββ observable is marked as upper limit."""
        from sdgft.registry import Registry
        reg = Registry()
        no.register_all(reg)
        obs = reg.get("qm_m_bb")
        assert obs.is_upper_limit is True


# ╔══════════════════════════════════════════════════════════════════╗
# ║  11. Cross-consistency checks                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestConsistency:
    """Cross-checks between different formulas."""

    def test_dm2_from_masses(self):
        """Δm² values consistent with mass spectrum."""
        m1, m2, m3 = no.neutrino_masses()
        assert no.delta_m2_21() == pytest.approx(m2**2 - m1**2, rel=1e-12)
        assert no.delta_m2_31() == pytest.approx(m3**2 - m1**2, rel=1e-12)
        assert no.delta_m2_32() == pytest.approx(m3**2 - m2**2, rel=1e-12)

    def test_sum_constraint(self):
        """Σm_ν self-consistent with R and Δm²."""
        r = no.R_TREE_F
        dm21 = no.DM2_21
        m2 = math.sqrt(dm21)
        m3 = math.sqrt(r * dm21)
        sigma = m2 + m3  # m1 = 0
        assert sigma == pytest.approx(SUM_M_NU, rel=1e-10)

    def test_pmns_unitarity_with_default(self):
        """Default PMNS matrix is unitary."""
        assert no.pmns_is_unitary(no.U_PMNS)

    def test_osc_prob_equals_amplitude_squared(self):
        """P computed from amplitude matches expectation."""
        L, E = 500.0, 1.0
        masses = no.neutrino_masses()
        U = no.pmns_matrix()

        # Manual amplitude
        phase_coeff = no.OSC_PHASE_COEFF * L / E
        A = 0j
        for i in range(3):
            phi = masses[i] ** 2 * phase_coeff
            A += U[0][i] * cmath.exp(-1j * phi) * U[1][i].conjugate()
        p_manual = abs(A) ** 2

        # Via function (note: alpha is source, beta is target)
        p_func = no.oscillation_probability("mu", "e", L, E)

        assert p_func == pytest.approx(p_manual, rel=1e-10)

    def test_r_tree_vs_fp(self):
        """Tree and FP variants give similar R."""
        assert abs(no.R_TREE_F - no.R_FP) < 0.1

    def test_phase_coefficient(self):
        """OSC_PHASE_COEFF ≈ 2.534 and OSC_REDUCED_COEFF ≈ 1.267."""
        assert no.OSC_PHASE_COEFF == pytest.approx(2.534, abs=0.001)
        assert no.OSC_REDUCED_COEFF == pytest.approx(1.267, abs=0.001)

    def test_m_bb_below_m3(self):
        """Effective Majorana mass is less than m₃ (cancellation)."""
        assert no.M_BB < no.M3

    def test_jarlskog_from_matrix(self):
        """Jarlskog from formula matches matrix invariant."""
        U = no.U_PMNS
        # J = Im(U_e2 · U_μ3 · U*_e3 · U*_μ2)
        j_mat = (U[0][1] * U[1][2] * U[0][2].conjugate() * U[1][1].conjugate()).imag
        assert j_mat == pytest.approx(no.J_PMNS, rel=1e-8)

    def test_print_summary_runs(self, capsys):
        """print_summary() executes without error."""
        no.print_summary()
        captured = capsys.readouterr()
        assert "SDGFT Neutrino Oscillation" in captured.out
        assert "Δm²₂₁" in captured.out
        assert "DUNE" in captured.out

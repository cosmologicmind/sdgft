"""Cross-module consistency checks for the SDGFT framework.

Validates that all algebraic identities, exact constraints, and
numerical convergences hold simultaneously. This is the theory's
internal self-test: if any check fails, there is a bug in the code
or in the axioms themselves.

Usage:
    from sdgft import validate
    validate()          # raises AssertionError on failure
    validate(loud=True) # prints each check as it passes
"""

from __future__ import annotations

from fractions import Fraction
import math

from .constants import DELTA, DELTA_G, DELTA_F, DELTA_G_F, PHI, SIN2_30, COS2_30
from .dimension import (
    D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP,
    N_TREE, N_FP,
    TWO_N_MINUS_1_TREE,
    compute_d_star_fp,
)
from .gravity import (
    ALPHA_M_TREE, ALPHA_B_TREE,
    ALPHA_M_FP, ALPHA_B_FP,
    ALPHA_T, ALPHA_K,
    alpha_m, alpha_b,
)
from .cosmology import (
    OMEGA_B, OMEGA_C, OMEGA_DE, OMEGA_M,
    W_DE_TREE, ETA_B,
    omega_b_formula, w_de,
)
from .inflation import (
    N_EFOLDS_FP, N_S, R_TENSOR, BETA_ISO, BETA_ISO_F,
    e_folds, spectral_index, tensor_to_scalar,
)
from .particle import (
    ALPHA_EM_INV_TREE, ALPHA_EM_INV_FP, ALPHA_S,
    SIN2_THETA_W, MU_E_RATIO, N_GEN, HIGGS_MASS,
    alpha_em_inv, alpha_s_value, sin2_theta_w,
)


class ValidationResult:
    """Collects pass/fail results from validation checks."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def check(self, condition: bool, name: str, detail: str = "") -> None:
        """Record a check result."""
        if condition:
            self.passed.append(name)
        else:
            self.failed.append((name, detail))

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        lines = [f"SDGFT Validation: {len(self.passed)} passed, {len(self.failed)} failed"]
        lines.append("-" * 60)
        for name in self.passed:
            lines.append(f"  PASS  {name}")
        for name, detail in self.failed:
            lines.append(f"  FAIL  {name}: {detail}")
        lines.append("-" * 60)
        if self.ok:
            lines.append("All checks passed.")
        else:
            lines.append(f"{len(self.failed)} check(s) FAILED.")
        return "\n".join(lines)


# ── Individual check functions ────────────────────────────────────


def check_axioms(r: ValidationResult) -> None:
    """Level 0: verify the two axioms and their relations."""
    r.check(
        DELTA == Fraction(5, 24),
        "axiom_delta",
        f"Delta = {DELTA}, expected 5/24",
    )
    r.check(
        DELTA_G == Fraction(1, 24),
        "axiom_delta_g",
        f"delta = {DELTA_G}, expected 1/24",
    )
    r.check(
        DELTA + DELTA_G == SIN2_30,
        "delta_sum",
        f"Delta + delta = {DELTA + DELTA_G}, expected {SIN2_30}",
    )
    r.check(
        DELTA / DELTA_G == 5,
        "delta_ratio",
        f"Delta / delta = {DELTA / DELTA_G}, expected 5",
    )
    r.check(
        SIN2_30 + COS2_30 == 1,
        "sin2_cos2_sum",
        f"sin^2 + cos^2 = {SIN2_30 + COS2_30}",
    )
    r.check(
        abs(PHI ** 2 - PHI - 1.0) < 1e-14,
        "golden_ratio_identity",
        f"phi^2 - phi - 1 = {PHI**2 - PHI - 1}",
    )


def check_dimension(r: ValidationResult) -> None:
    """Level 2: D* tree formula and fixed-point convergence."""
    r.check(
        D_STAR_TREE == 3 - SIN2_30 + DELTA_G,
        "d_star_tree_formula_a",
        f"3 - sin^2(30) + delta = {3 - SIN2_30 + DELTA_G}, expected {D_STAR_TREE}",
    )
    r.check(
        D_STAR_TREE == 3 - DELTA,
        "d_star_tree_formula_b",
        f"3 - Delta = {3 - DELTA}, expected {D_STAR_TREE}",
    )
    r.check(
        N_TREE == D_STAR_TREE / 2,
        "n_tree_is_half_d_star",
        f"D*/2 = {D_STAR_TREE / 2}, expected {N_TREE}",
    )
    r.check(
        TWO_N_MINUS_1_TREE == 2 * N_TREE - 1,
        "two_n_minus_1_tree",
        f"2n-1 = {2 * N_TREE - 1}, expected {TWO_N_MINUS_1_TREE}",
    )
    # Fixed-point self-consistency
    correction = DELTA_F ** (DELTA_F * DELTA_G_F)
    fp_check = DELTA_F ** (-1.0 / D_STAR_FP) * PHI * correction
    r.check(
        abs(fp_check - D_STAR_FP) < 1e-12,
        "fp_self_consistency",
        f"f(D*_fp) = {fp_check}, D*_fp = {D_STAR_FP}, diff = {abs(fp_check - D_STAR_FP)}",
    )
    # FP converges from different starting points
    for d0 in [1.5, 2.0, 3.0, 4.0, 5.0]:
        val, _ = compute_d_star_fp(d0=d0)
        r.check(
            abs(val - D_STAR_FP) < 1e-12,
            f"fp_convergence_d0={d0}",
            f"from d0={d0}: {val} vs {D_STAR_FP}",
        )
    # Tree and FP agree within 1%
    pct_diff = abs(D_STAR_TREE_F - D_STAR_FP) / D_STAR_FP * 100
    r.check(
        pct_diff < 1.0,
        "tree_fp_agreement",
        f"difference = {pct_diff:.4f}%",
    )


def check_gravity(r: ValidationResult) -> None:
    """Level 3: Horndeski parameter identities."""
    r.check(
        ALPHA_M_TREE == (N_TREE - 1) / (2 * N_TREE - 1),
        "alpha_m_tree_fraction",
        f"{(N_TREE - 1) / (2 * N_TREE - 1)} vs {ALPHA_M_TREE}",
    )
    r.check(
        ALPHA_B_TREE == -ALPHA_M_TREE / 2,
        "alpha_b_is_neg_half_alpha_m",
        f"{-ALPHA_M_TREE / 2} vs {ALPHA_B_TREE}",
    )
    r.check(
        ALPHA_T == 0,
        "alpha_t_zero",
        f"alpha_T = {ALPHA_T}",
    )
    r.check(
        ALPHA_K == 0,
        "alpha_k_zero",
        f"alpha_K = {ALPHA_K}",
    )
    # Function consistency
    r.check(
        abs(alpha_m(float(N_TREE)) - float(ALPHA_M_TREE)) < 1e-14,
        "alpha_m_function_vs_constant",
    )
    r.check(
        abs(alpha_b(float(N_TREE)) - float(ALPHA_B_TREE)) < 1e-14,
        "alpha_b_function_vs_constant",
    )


def check_cosmology(r: ValidationResult) -> None:
    """Level 5-6: Exact flatness and density constraints."""
    r.check(
        OMEGA_B + OMEGA_C + OMEGA_DE == 1,
        "exact_flatness",
        f"sum = {OMEGA_B + OMEGA_C + OMEGA_DE}",
    )
    r.check(
        OMEGA_M == OMEGA_B + OMEGA_C,
        "omega_m_is_sum",
        f"Omega_b + Omega_c = {OMEGA_B + OMEGA_C}, expected {OMEGA_M}",
    )
    r.check(
        OMEGA_B == omega_b_formula(DELTA, DELTA_G),
        "omega_b_formula",
        f"formula gives {omega_b_formula(DELTA, DELTA_G)}, expected {OMEGA_B}",
    )
    r.check(
        W_DE_TREE == -D_STAR_TREE / 3,
        "w_de_tree_fraction",
        f"-D*/3 = {-D_STAR_TREE / 3}, expected {W_DE_TREE}",
    )
    # w_DE is quintessence-like (> -1)
    r.check(
        float(W_DE_TREE) > -1.0,
        "w_de_quintessence",
        f"w_DE = {float(W_DE_TREE)}, should be > -1",
    )
    # eta_B order of magnitude
    r.check(
        1e-11 < ETA_B < 1e-8,
        "eta_b_order_of_magnitude",
        f"eta_B = {ETA_B}",
    )


def check_inflation(r: ValidationResult) -> None:
    """Level 4: Inflationary consistency relations."""
    # N_e in reasonable range
    r.check(
        50.0 < N_EFOLDS_FP < 70.0,
        "n_efolds_range",
        f"N_e = {N_EFOLDS_FP}",
    )
    # n_s < 1 (red tilt)
    r.check(
        N_S < 1.0,
        "n_s_red_tilt",
        f"n_s = {N_S}",
    )
    # r > 0
    r.check(
        R_TENSOR > 0.0,
        "r_positive",
        f"r = {R_TENSOR}",
    )
    # r < BICEP2/Keck upper limit
    r.check(
        R_TENSOR < 0.036,
        "r_below_bicep",
        f"r = {R_TENSOR}",
    )
    # beta_iso < Planck limit
    r.check(
        BETA_ISO_F < 0.038,
        "beta_iso_below_planck",
        f"beta_iso = {BETA_ISO_F}",
    )
    # f(R) consistency: r = 12*(1 - n_s)^2 exactly for R^n inflation.
    # Proof: both r and (1-n_s) share the denominator D = N_e*(2n-1) + n,
    # so r = 48*(2n-1)^2/D^2 = 12 * [2*(2n-1)/D]^2 = 12*(1-n_s)^2.
    r_exact = 12.0 * (1.0 - N_S) ** 2
    r.check(
        abs(R_TENSOR - r_exact) / R_TENSOR < 1e-12,
        "r_fR_consistency",
        f"r = {R_TENSOR}, 12*(1-n_s)^2 = {r_exact}",
    )


def check_particle(r: ValidationResult) -> None:
    """Level 5-6: Particle physics internal consistency."""
    # alpha_em_inv is positive and near 137
    r.check(
        130.0 < ALPHA_EM_INV_TREE < 140.0,
        "alpha_em_inv_range",
        f"alpha_em^-1 = {ALPHA_EM_INV_TREE}",
    )
    # alpha_s is positive and ~ 0.118
    r.check(
        0.10 < ALPHA_S < 0.13,
        "alpha_s_range",
        f"alpha_s = {ALPHA_S}",
    )
    # sin2_theta_w in range
    r.check(
        0.22 < SIN2_THETA_W < 0.24,
        "sin2_theta_w_range",
        f"sin^2(theta_W) = {SIN2_THETA_W}",
    )
    # Exactly 3 generations
    r.check(
        N_GEN == 3,
        "n_generations_is_3",
        f"N_gen = {N_GEN}",
    )
    # Higgs mass near 125 GeV
    r.check(
        120.0 < HIGGS_MASS < 130.0,
        "higgs_mass_range",
        f"m_H = {HIGGS_MASS}",
    )
    # mu/e ratio positive and near 207
    r.check(
        200.0 < MU_E_RATIO < 210.0,
        "mu_e_ratio_range",
        f"m_mu/m_e = {MU_E_RATIO}",
    )
    # Tree and FP alpha_em are close
    pct = abs(ALPHA_EM_INV_TREE - ALPHA_EM_INV_FP) / ALPHA_EM_INV_FP * 100
    r.check(
        pct < 1.0,
        "alpha_em_tree_vs_fp",
        f"difference = {pct:.4f}%",
    )


# ── Public API ────────────────────────────────────────────────────


def check_all() -> ValidationResult:
    """Run all consistency checks and return a ValidationResult."""
    result = ValidationResult()
    check_axioms(result)
    check_dimension(result)
    check_gravity(result)
    check_cosmology(result)
    check_inflation(result)
    check_particle(result)
    return result


def validate(loud: bool = False) -> None:
    """Run all checks; raise AssertionError if any fail.

    Args:
        loud: If True, print every check as it passes.
    """
    result = check_all()
    if loud:
        print(result.summary())
    if not result.ok:
        msg = "\n".join(f"  {name}: {detail}" for name, detail in result.failed)
        raise AssertionError(f"SDGFT validation failed:\n{msg}")

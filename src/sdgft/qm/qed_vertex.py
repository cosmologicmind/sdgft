r"""QED vertex corrections: anomalous magnetic moments (g−2) in SDGFT.

SDGFT predicts a geometric correction to the lepton anomalous magnetic
moments arising from the 24-cell topology, proportional to the SQUARE
of the geometric anomalous dimension γ_geo = δ²/D*.

============================================================
Physical Picture
============================================================

In the SDGFT vacuum, the 24-cell lattice structure introduces a
geometric anomalous dimension

    γ_geo = δ²/D* = (1/24)² / (67/24) = 1 / (24 × 67) = 1/1608

into the RG running of all fields.  This is EXACTLY the same anomalous
dimension that drives the dark energy RG flow:

    Ω_DE(r) = (3/4) · (r/r_P)^{−γ_geo}

For the QED vertex function, the geometric anomalous dimension modifies
the one-loop Schwinger triangle diagram.  The standard Schwinger result
a^{(1)} = α/(2π) assumes flat ℝ^{3,1}.  In the SDGFT cone geometry,
the virtual-photon loop picks up corrections from the lattice tension δ.

============================================================
Why γ²_geo (squared)?
============================================================

The anomalous magnetic moment a = F₂(0) is a chirality-FLIPPING
operator (proportional to σ^{μν}, mixing left- and right-handed spinors).
At leading order, the geometric anomalous dimension γ_geo is chirality-
PRESERVING: it uniformly rescales propagators via the Ward identity,
modifying only the charge form factor F₁, not the magnetic moment F₂.

The chirality-flipping component enters at SECOND order:

    δF₂ ∝ ⟨L|H²_geo|R⟩ ∝ γ²_geo

This parallels the standard result: the magnetic moment is a one-loop
effect (∝ α) while charge renormalisation is tree-level (∝ α⁰).

============================================================
The Formula
============================================================

The SDGFT geometric correction to the anomalous magnetic moment of
a lepton with mass m_ℓ is:

    Δa_ℓ^{geo} = (α/2π) · γ²_geo · ln(m_ℓ/m_e)

              = (α/2π) · (δ²/D*)² · ln(m_ℓ/m_e)

              = (α/2π) · 1/1608² · ln(m_ℓ/m_e)

This formula has ZERO free parameters — everything derives from:
    δ  = 1/24    (elementary lattice tension, 24-cell axiom)
    D* = 67/24   (effective dimension, Level 2)
    α             (fine-structure constant, predicted at Level 5)
    m_ℓ/m_e       (lepton mass ratio, predicted at Level 5)

The lower integration limit m_e (electron mass) is the lightest
charged-fermion scale: below m_e there are no virtual fermion
contributions to the geometric RG running.

============================================================
Predictions
============================================================

Electron (reference, no running below m_e):
    Δa_e^{geo} = 0  (exactly)

Muon:
    Δa_μ^{geo} = (α/2π) · (1/1608)² · ln(m_μ/m_e)  ≈ 2.39 × 10⁻⁹
    Observed anomaly (2020 WP):  (2.49 ± 0.48) × 10⁻⁹  →  0.2σ

Tau (prediction for future experiments):
    Δa_τ^{geo} = (α/2π) · (1/1608)² · ln(m_τ/m_e)  ≈ 3.66 × 10⁻⁹

============================================================
Connection to Dark Energy
============================================================

The geometricanomalous dimension γ_geo = δ²/D* = γ_DE is IDENTICAL
to the quantity governing the dark energy RG flow.  Both originate
from the same 24-cell lattice tension δ acting on D*-dimensional
geometry.  The dark energy running and the muon g-2 anomaly are
two manifestations of the same topological structure:

    Cosmology:  Ω_DE ∝ r^{−γ_geo}
    QED vertex: Δa_μ ∝ γ²_geo · ln(m_μ/m_e)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ..constants import DELTA_G, DELTA_G_F
from ..dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from ..dimensional_flow import GAMMA_DE_TREE, GAMMA_DE_TREE_F
from ..particle import (
    ALPHA_EM_TREE,
    ALPHA_EM_INV_TREE,
    MU_E_RATIO,
    TAU_MU_RATIO_TREE,
    TAU_E_RATIO_TREE,
)
from ..registry import Observable, REGISTRY


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Physical constants (PDG 2024 / CODATA 2022)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

ALPHA_INV_OBS: float = 137.035999177
"""Observed inverse fine-structure constant (2022 CODATA)."""

ALPHA_OBS: float = 1.0 / ALPHA_INV_OBS
"""Observed fine-structure constant: ~7.297 × 10⁻³."""

M_E_MEV: float = 0.51099895000
"""Electron mass [MeV] (PDG 2024)."""

M_MU_MEV: float = 105.6583755
"""Muon mass [MeV] (PDG 2024)."""

M_TAU_MEV: float = 1776.86
"""Tau mass [MeV] (PDG 2024)."""

M_MU_OVER_M_E: float = 206.7682830
"""Muon-to-electron mass ratio (PDG 2024, direct measurement)."""

M_TAU_OVER_M_E: float = 3477.48
"""Tau-to-electron mass ratio (PDG 2024)."""


# ── Experimental g-2 values ───────────────────────────────────────

A_MU_EXP: float = 116_592_059e-11
"""Muon g-2 experimental world average (BNL + Fermilab Run 1-3).

    a_μ^{exp} = 0.00116592059(22)
    Phys. Rev. Lett. 131, 161802 (2023)
"""

A_MU_EXP_UNCERT: float = 22e-11
"""1σ uncertainty on a_μ^{exp}."""

A_E_EXP: float = 0.001_159_652_180_59
"""Electron g-2 experimental value.

    a_e^{exp} = 0.00115965218059(13)
    Fan et al., Phys. Rev. Lett. 130, 071801 (2023)
"""

A_E_EXP_UNCERT: float = 1.3e-13
"""1σ uncertainty on a_e^{exp}."""


# ── SM predictions ────────────────────────────────────────────────

A_MU_SM_WP: float = 116_591_810e-11
"""SM prediction for muon g-2 (2020 Theory Initiative White Paper).

    a_μ^{SM} = 0.00116591810(43)
    T. Aoyama et al., Phys. Rept. 887 (2020) 1
    Based on data-driven hadronic vacuum polarisation (HVP).
"""

A_MU_SM_WP_UNCERT: float = 43e-11
"""1σ uncertainty on a_μ^{SM} (WP)."""

DELTA_A_MU_OBS: float = A_MU_EXP - A_MU_SM_WP
"""Observed muon g-2 anomaly (WP basis): ~2.49 × 10⁻⁹, 5.1σ."""

DELTA_A_MU_OBS_UNCERT: float = math.sqrt(
    A_MU_EXP_UNCERT ** 2 + A_MU_SM_WP_UNCERT ** 2
)
"""Combined 1σ uncertainty on the anomaly: ~4.8 × 10⁻¹⁰."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Geometric anomalous dimension                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def gamma_geo_exact() -> Fraction:
    """Geometric anomalous dimension (exact rational).

    γ_geo = δ²/D* = (1/24)² / (67/24) = 1/(24 × 67) = 1/1608

    The number 1608 = 24 × 67 encodes both:
        24  = vertex count of the 24-cell
        67  = numerator of D*_tree = 67/24

    IDENTICAL to the dark energy anomalous dimension γ_DE
    from the dimensional_flow module.
    """
    return DELTA_G ** 2 / D_STAR_TREE


GAMMA_GEO: Fraction = gamma_geo_exact()
"""γ_geo = 1/1608 (exact rational)."""

GAMMA_GEO_F: float = float(GAMMA_GEO)
"""Float alias: ≈ 6.2189 × 10⁻⁴."""

GAMMA_GEO_SQ: Fraction = GAMMA_GEO ** 2
"""γ²_geo = 1/2585664 (exact rational)."""

GAMMA_GEO_SQ_F: float = float(GAMMA_GEO_SQ)
"""Float alias: ≈ 3.8675 × 10⁻⁷."""


# ── Cross-checks ──────────────────────────────────────────────────

assert GAMMA_GEO == GAMMA_DE_TREE, (
    f"γ_geo = {GAMMA_GEO} must equal γ_DE = {GAMMA_DE_TREE}"
)

assert GAMMA_GEO == Fraction(1, 1608), (
    f"γ_geo = {GAMMA_GEO}, expected 1/1608"
)

assert GAMMA_GEO_SQ == Fraction(1, 2_585_664), (
    f"γ²_geo = {GAMMA_GEO_SQ}, expected 1/2585664"
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SDGFT geometric vertex correction                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def delta_a_lepton(
    m_ell_over_m_e: float,
    alpha: float = ALPHA_OBS,
    gamma_sq: float = GAMMA_GEO_SQ_F,
) -> float:
    """SDGFT geometric correction to the lepton anomalous magnetic moment.

    Δa_ℓ^{geo} = (α/2π) · γ²_geo · ln(m_ℓ/m_e)

    Physical basis:
        1. The anomalous magnetic moment is chirality-flipping (σ^{μν}).
        2. The geometric anomalous dimension γ_geo = δ²/D* is chirality-
           preserving at leading order (Ward identity → only F₁).
        3. The chirality-flip enters at SECOND order: F₂ ∝ γ²_geo.
        4. The log factor arises from RG running between m_e (lightest
           charged fermion) and m_ℓ.

    Args:
        m_ell_over_m_e: Lepton-to-electron mass ratio.
        alpha: Fine-structure constant (default: observed CODATA value).
        gamma_sq: Square of geometric anomalous dimension (default: 1/1608²).

    Returns:
        Δa_ℓ^{geo} (dimensionless).  Returns 0.0 for m_ℓ ≤ m_e.
    """
    if m_ell_over_m_e <= 1.0:
        return 0.0
    return (alpha / (2.0 * math.pi)) * gamma_sq * math.log(m_ell_over_m_e)


def delta_a_electron() -> float:
    """SDGFT geometric correction to electron g-2: identically zero.

    The electron is the lightest charged fermion — there is no
    geometric RG running below m_e, so ln(m_e/m_e) = 0.

    This is crucial: the electron g-2 is the single most precisely
    measured quantity in physics (0.13 ppt), and any BSM correction
    must be negligibly small.  SDGFT naturally gives EXACTLY zero.
    """
    return 0.0


def delta_a_muon(
    alpha: float = ALPHA_OBS,
    gamma_sq: float = GAMMA_GEO_SQ_F,
    mass_ratio: float = M_MU_OVER_M_E,
) -> float:
    """SDGFT geometric correction to muon g-2.

    Δa_μ^{geo} = (α/2π) · (1/1608)² · ln(m_μ/m_e) ≈ 2.39 × 10⁻⁹

    This predicts the observed muon g-2 anomaly to within 0.2σ.
    """
    return delta_a_lepton(mass_ratio, alpha, gamma_sq)


def delta_a_tau(
    alpha: float = ALPHA_OBS,
    gamma_sq: float = GAMMA_GEO_SQ_F,
    mass_ratio: float = M_TAU_OVER_M_E,
) -> float:
    """SDGFT geometric correction to tau g-2.

    Δa_τ^{geo} = (α/2π) · (1/1608)² · ln(m_τ/m_e) ≈ 3.66 × 10⁻⁹

    Prediction for future experiments (Belle II, FCC-ee).
    Currently unmeasurable: best bound |a_τ| < 0.013 (95% CL).
    """
    return delta_a_lepton(mass_ratio, alpha, gamma_sq)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Schwinger term (leading QED)                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def schwinger_term(alpha: float = ALPHA_OBS) -> float:
    """Schwinger one-loop anomalous magnetic moment: a^{(1)} = α/(2π).

    This is the STANDARD QED result, identical in SDGFT and SM when
    using the same α.  The SDGFT modification enters separately
    through the geometric vertex correction Δa_ℓ^{geo}.
    """
    return alpha / (2.0 * math.pi)


def schwinger_sdgft() -> float:
    """Schwinger term using SDGFT tree-level alpha.

    a^{(1)}_SDGFT = α_tree/(2π)

    Uses α^{-1}_tree = 2πD*³ + δD* ≈ 136.82.
    """
    return schwinger_term(alpha=ALPHA_EM_TREE)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  d-dimensional Schwinger function Ξ(d) — Diagnostic only       ║
# ╚══════════════════════════════════════════════════════════════════╝

def xi_d(d: float) -> float:
    """Dimension-dependent Schwinger enhancement factor Ξ(d).

    In d spacetime dimensions, the one-loop anomalous magnetic moment is

        a^{(1)}(d) = (α/2π) · Ξ(d)

    with Ξ(4) = 1 (standard QED) and

        Ξ(d) = (d−2)·Γ(3−d/2)·[Γ(d/2−1)]²·(4π)^{2−d/2}  /  [2·Γ(d−2)]

    ⚠️  DIAGNOSTIC ONLY.  This is NOT the physical SDGFT prediction.

    The physical SDGFT correction enters perturbatively through
    γ²_geo (see delta_a_lepton), not through a naive change of the
    loop-integral dimension.  At d_eff = 1+D* ≈ 3.79, Ξ ≈ 1.37,
    which would change the Schwinger term by 37% — far too large.
    The actual SDGFT correction is ~10⁻⁶ relative.

    Args:
        d: Spacetime dimension (must be > 2).

    Returns:
        Ξ(d) enhancement factor.
    """
    if d <= 2.0:
        return 0.0
    return (
        (d - 2.0)
        * math.gamma(3.0 - d / 2.0)
        * math.gamma(d / 2.0 - 1.0) ** 2
        * (4.0 * math.pi) ** (2.0 - d / 2.0)
        / (2.0 * math.gamma(d - 2.0))
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Module-level computed constants                                ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Geometric corrections (using observed α & PDG masses) ─────────

DELTA_A_E: float = delta_a_electron()
"""Geometric correction to electron g-2: exactly 0."""

DELTA_A_MU: float = delta_a_muon()
"""Geometric correction to muon g-2: ~2.39 × 10⁻⁹."""

DELTA_A_TAU: float = delta_a_tau()
"""Geometric correction to tau g-2: ~3.66 × 10⁻⁹."""


# ── Pure SDGFT predictions (using SDGFT α and mass ratios) ───────

DELTA_A_MU_PURE: float = delta_a_muon(
    alpha=ALPHA_EM_TREE,
    mass_ratio=MU_E_RATIO,
)
"""Pure SDGFT muon correction (tree α, SDGFT m_μ/m_e)."""

DELTA_A_TAU_PURE: float = delta_a_tau(
    alpha=ALPHA_EM_TREE,
    mass_ratio=TAU_E_RATIO_TREE,
)
"""Pure SDGFT tau correction (tree α, SDGFT m_τ/m_e)."""


# ── Total SDGFT prediction for a_μ (SM + geometric) ──────────────

A_MU_SDGFT_TOTAL: float = A_MU_SM_WP + DELTA_A_MU
"""Total SDGFT prediction: a_μ^{SM} + Δa_μ^{geo}."""

A_MU_SDGFT_SIGMA: float = (
    abs(A_MU_SDGFT_TOTAL - A_MU_EXP)
    / math.sqrt(A_MU_EXP_UNCERT ** 2 + A_MU_SM_WP_UNCERT ** 2)
)
"""Tension of SDGFT total prediction with experiment [σ]."""


# ── Schwinger terms ───────────────────────────────────────────────

SCHWINGER_OBS: float = schwinger_term()
"""α_obs/(2π) ≈ 1.1614 × 10⁻³."""

SCHWINGER_SDGFT: float = schwinger_sdgft()
"""α_tree/(2π) — Schwinger term from SDGFT alpha."""


# ── d-dimensional diagnostic ─────────────────────────────────────

D_EFF: float = 1.0 + D_STAR_TREE_F
"""Effective spacetime dimension: 1 + D* = 91/24 ≈ 3.7917."""

XI_AT_D_EFF: float = xi_d(D_EFF)
"""Ξ(1+D*) ≈ 1.37 — diagnostic only, NOT the physical SDGFT correction."""


# ── Lepton correction ratio (mass-independent prediction) ────────

TAU_MU_GEO_RATIO: float = (
    math.log(M_TAU_OVER_M_E) / math.log(M_MU_OVER_M_E)
)
"""Δa_τ/Δa_μ = ln(m_τ/m_e)/ln(m_μ/m_e) ≈ 1.529.

A pure geometric prediction independent of α and γ_geo.
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Experiment predictions                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class G2Prediction:
    """SDGFT g-2 prediction for a specific lepton."""

    lepton: str
    mass_mev: float
    mass_ratio: float
    delta_a_geo: float
    a_sm: float | None
    a_exp: float | None
    a_exp_uncert: float | None

    @property
    def a_sdgft(self) -> float | None:
        """Total SDGFT prediction: a_SM + Δa_geo."""
        if self.a_sm is None:
            return None
        return self.a_sm + self.delta_a_geo

    @property
    def sigma_vs_exp(self) -> float | None:
        """Tension between SDGFT and experiment [σ]."""
        a = self.a_sdgft
        if a is None or self.a_exp is None or self.a_exp_uncert is None:
            return None
        if self.a_exp_uncert <= 0.0:
            return None
        # include SM theory uncertainty for muon
        if self.lepton == "μ" and A_MU_SM_WP_UNCERT > 0:
            combined = math.sqrt(
                self.a_exp_uncert ** 2 + A_MU_SM_WP_UNCERT ** 2
            )
            return abs(a - self.a_exp) / combined
        return abs(a - self.a_exp) / self.a_exp_uncert

    @property
    def fraction_of_anomaly(self) -> float | None:
        """Fraction of observed anomaly explained by SDGFT."""
        if self.a_sm is None or self.a_exp is None:
            return None
        anomaly = self.a_exp - self.a_sm
        if anomaly == 0.0:
            return None
        return self.delta_a_geo / anomaly


def predict_electron() -> G2Prediction:
    """SDGFT prediction for electron g-2."""
    return G2Prediction(
        lepton="e",
        mass_mev=M_E_MEV,
        mass_ratio=1.0,
        delta_a_geo=DELTA_A_E,
        a_sm=None,  # SM a_e depends on which α is used
        a_exp=A_E_EXP,
        a_exp_uncert=A_E_EXP_UNCERT,
    )


def predict_muon() -> G2Prediction:
    """SDGFT prediction for muon g-2."""
    return G2Prediction(
        lepton="μ",
        mass_mev=M_MU_MEV,
        mass_ratio=M_MU_OVER_M_E,
        delta_a_geo=DELTA_A_MU,
        a_sm=A_MU_SM_WP,
        a_exp=A_MU_EXP,
        a_exp_uncert=A_MU_EXP_UNCERT,
    )


def predict_tau() -> G2Prediction:
    """SDGFT prediction for tau g-2 (future)."""
    return G2Prediction(
        lepton="τ",
        mass_mev=M_TAU_MEV,
        mass_ratio=M_TAU_OVER_M_E,
        delta_a_geo=DELTA_A_TAU,
        a_sm=None,  # SM HVP for tau not yet at needed precision
        a_exp=None,  # not measurable with current technology
        a_exp_uncert=None,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Registry                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def register_all(registry=REGISTRY) -> None:
    """Register all QED vertex observables."""

    # Primary result: geometric correction to muon g-2
    registry.register(Observable(
        name="qm_delta_a_mu_geo",
        symbol="Δa_μ^{geo}",
        formula="(α/2π)·(δ²/D*)²·ln(m_μ/m_e)",
        predicted=DELTA_A_MU,
        observed=DELTA_A_MU_OBS,
        observed_uncertainty=DELTA_A_MU_OBS_UNCERT,
        unit="dimensionless",
        level=7,
        d_star_variant="tree",
        dependencies=("delta_g", "d_star_tree", "alpha_em_inv", "mu_e_ratio"),
    ))

    # Total SDGFT prediction for a_mu (SM + geometric)
    registry.register(Observable(
        name="qm_a_mu_total",
        symbol="a_μ^{SDGFT}",
        formula="a_μ^{SM} + Δa_μ^{geo}",
        predicted=A_MU_SDGFT_TOTAL,
        observed=A_MU_EXP,
        observed_uncertainty=math.sqrt(
            A_MU_EXP_UNCERT ** 2 + A_MU_SM_WP_UNCERT ** 2
        ),
        unit="dimensionless",
        level=7,
        d_star_variant="tree",
        dependencies=("qm_delta_a_mu_geo",),
    ))

    # Tau prediction (future)
    registry.register(Observable(
        name="qm_delta_a_tau_geo",
        symbol="Δa_τ^{geo}",
        formula="(α/2π)·(δ²/D*)²·ln(m_τ/m_e)",
        predicted=DELTA_A_TAU,
        observed=float("nan"),
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=7,
        d_star_variant="tree",
        dependencies=("delta_g", "d_star_tree", "alpha_em_inv"),
    ))

    # Geometric anomalous dimension (diagnostic cross-check with γ_DE)
    registry.register(Observable(
        name="qm_gamma_geo",
        symbol="γ_geo",
        formula="δ²/D* = 1/1608",
        predicted=GAMMA_GEO_F,
        observed=GAMMA_DE_TREE_F,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=7,
        d_star_variant="tree",
        dependencies=("delta_g", "d_star_tree"),
        is_diagnostic=True,
    ))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Summary printer                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def print_summary() -> None:
    """Print formatted summary of SDGFT g-2 predictions."""
    sep = "=" * 76
    print(sep)
    print("  SDGFT QED Vertex Corrections — Anomalous Magnetic Moments (g−2)")
    print("  Zero Free Parameters: δ = 1/24, D* = 67/24")
    print(sep)

    # Geometric anomalous dimension
    print()
    print("  Geometric anomalous dimension:")
    print(f"    γ_geo = δ²/D* = 1/1608 = {GAMMA_GEO_F:.6e}")
    print(f"    γ²_geo = 1/2585664 = {GAMMA_GEO_SQ_F:.6e}")
    print(f"    Cross-check: γ_geo ≡ γ_DE (dark energy) ✓")

    # Formula
    print()
    print("  Formula:")
    print("    Δa_ℓ = (α/2π) · γ²_geo · ln(m_ℓ/m_e)")

    # Lepton corrections
    print()
    print("  Lepton corrections (using observed α, PDG masses):")
    print(f"    Δa_e  = {DELTA_A_E:12.4e}  (exactly zero, electron = reference)")
    print(f"    Δa_μ  = {DELTA_A_MU:12.4e}  (obs anomaly: "
          f"{DELTA_A_MU_OBS:.2e} ± {DELTA_A_MU_OBS_UNCERT:.2e}, "
          f"{abs(DELTA_A_MU - DELTA_A_MU_OBS) / DELTA_A_MU_OBS_UNCERT:.2f}σ)")
    print(f"    Δa_τ  = {DELTA_A_TAU:12.4e}  (prediction, no data)")

    # Tau/muon ratio
    print()
    print("  Lepton ratio (pure geometry, independent of α and γ_geo):")
    print(f"    Δa_τ/Δa_μ = ln(m_τ/m_e)/ln(m_μ/m_e) = {TAU_MU_GEO_RATIO:.4f}")

    # Pure SDGFT prediction
    print()
    print("  Pure SDGFT (tree α, SDGFT mass ratios):")
    print(f"    α⁻¹_tree = {ALPHA_EM_INV_TREE:.4f}")
    print(f"    m_μ/m_e  = {MU_E_RATIO:.4f}  (obs: {M_MU_OVER_M_E:.4f})")
    print(f"    Δa_μ     = {DELTA_A_MU_PURE:.4e}")
    print(f"    Δa_τ     = {DELTA_A_TAU_PURE:.4e}")

    # Total muon prediction
    mu = predict_muon()
    print()
    print("  Muon g-2 — Full comparison:")
    print(f"    a_μ^{{SM}}    = {A_MU_SM_WP:.11e}  (2020 WP)")
    print(f"    Δa_μ^{{geo}}  = {DELTA_A_MU:+.4e}")
    print(f"    a_μ^{{SDGFT}} = {A_MU_SDGFT_TOTAL:.11e}")
    print(f"    a_μ^{{exp}}   = {A_MU_EXP:.11e}")
    print(f"    Tension:     {A_MU_SDGFT_SIGMA:.2f}σ  "
          f"(SM alone: 5.1σ)")
    if mu.fraction_of_anomaly is not None:
        print(f"    SDGFT explains {mu.fraction_of_anomaly * 100:.1f}% "
              f"of the observed anomaly")

    # Schwinger terms
    print()
    print("  Schwinger term a^(1) = α/(2π):")
    print(f"    Observed α:  {SCHWINGER_OBS:.10e}")
    print(f"    SDGFT α:     {SCHWINGER_SDGFT:.10e}")
    print(f"    Difference:  {abs(SCHWINGER_OBS - SCHWINGER_SDGFT):.4e}  "
          f"(from tree-level α discrepancy)")

    # Diagnostic: d-dimensional Schwinger
    print()
    print("  Diagnostic — d-dimensional Schwinger Ξ(d):")
    print(f"    Ξ(4) = {xi_d(4.0):.6f}  (standard QED)")
    print(f"    Ξ(3) = {xi_d(3.0):.6f}  (QED₃)")
    print(f"    Ξ({D_EFF:.4f}) = {XI_AT_D_EFF:.6f}  (d = 1+D*)")
    print(f"    ⚠ Ξ(d) is diagnostic only — not the SDGFT prediction")

    print()
    print(sep)


if __name__ == "__main__":
    print_summary()

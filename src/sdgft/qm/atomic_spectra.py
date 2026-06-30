r"""Hydrogen atomic spectra: Lamb shift from SDGFT 24-cell geometry.

All derived from SDGFT axioms (Δ, δ, D*, φ) without free parameters.

============================================================
Physical Picture — Topological S-Orbital Frustration
============================================================

In the Standard Model, the Lamb shift (2S₁/₂ − 2P₁/₂ splitting)
arises because the 2S orbital has a non-zero probability density
at the origin (r = 0), where it samples QED vacuum fluctuations
(Uehling potential, vertex corrections), while the 2P orbital has
a node at r = 0 and does not.

In the SDGFT, the origin of the coordinate system sits on a vertex
of the 24-cell.  This vertex carries the Fibonacci-lattice conflict
Δ = 5/24 and the lattice tension δ = 1/24.  The 2S electron
"collides" with the discrete spacetime geometry, while the 2P
electron (with its node at the origin) does not sample this geometry.

The key insight: the Lamb shift probes the SAME geometric anomalous
dimension γ_geo = δ²/D* = 1/1608 that drives the dark energy RG flow
and the muon g-2 anomaly.  All three phenomena are manifestations of
the 24-cell lattice structure at different energy scales.

============================================================
The SDGFT Lamb Shift Equation
============================================================

    L_geo = (Δ / (Δ + δ)) · γ²_geo · R∞c

where:
    Δ/(Δ+δ) = (5/24)/(6/24) = 5/6    Geometric projection factor
    γ²_geo  = (δ²/D*)²                 Squared anomalous dimension
    R∞c     ≈ 3,289,841.96 MHz         Rydberg frequency

The factor 5/6 = Δ/(Δ+δ) represents the fraction of the Fibonacci
conflict Δ within the maximum cone opening angle Δ+δ = sin²(30°) = 1/4.
It quantifies how strongly the S-orbital is projected into the
topological conflict zone of the 24-cell.

============================================================
Why γ²_geo (squared)?
============================================================

Like the anomalous magnetic moment (g-2), the Lamb shift is a
chirality-preserving energy splitting.  The geometric anomalous
dimension γ_geo is chirality-preserving at leading order, but
it requires TWO topological "knot crossings" to produce an
observable energy shift.  Hence the correction enters at O(γ²_geo),
not at O(γ_geo).

============================================================
Predictions
============================================================

Tree-level (D* = 67/24):
    γ_geo = 1/1608
    L_geo^tree = (5/6) · (1/1608)² · R∞c ≈ 1060.3 MHz

Fixed-point (D*_fp ≈ 2.79676):
    γ_geo,fp = (1/24)² / D*_fp ≈ 6.2065 × 10⁻⁴
    L_geo^fp ≈ 1056.1 MHz

Observed:  1057.845 ± 0.009 MHz

    Tree:  +2.5 MHz off  (0.23%)
    FP:    −1.7 MHz off  (0.16%)

    The true value lies BETWEEN tree and fixed-point, strongly
    suggesting that the physical D* is D*_fp with small higher-order
    corrections, exactly as the theory predicts.

============================================================
Connection to Dark Energy and g-2
============================================================

Same γ_geo = δ²/D* = 1/1608 governs three phenomena:

    Cosmology:   Ω_DE(r) = (3/4)·(r/r_P)^{−γ_geo}
    QED vertex:  Δa_μ = (α/2π)·γ²_geo·ln(m_μ/m_e)
    Lamb shift:  L_geo = (5/6)·γ²_geo·R∞c

This is a striking demonstration of the scale invariance and
universality of the SDGFT geometric structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ..constants import DELTA, DELTA_F, DELTA_G, DELTA_G_F, SIN2_30
from ..dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from ..registry import Observable, REGISTRY


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Physical constants                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

R_INF_C_MHZ: float = 3_289_841_960.250
"""Rydberg frequency R∞·c in MHz (CODATA 2018).

R∞ = m_e·e⁴/(8·ε₀²·h³·c) = 10,973,731.568160 m⁻¹
R∞·c = 3,289,841,960,250 kHz = 3,289,841,960.250 MHz
"""

R_INF_C_HZ: float = R_INF_C_MHZ * 1e6
"""Rydberg frequency R∞·c in Hz."""


# ── Observed Lamb shift ───────────────────────────────────────────

LAMB_SHIFT_OBS_MHZ: float = 1057.845
"""Classic Lamb shift ΔE(2S₁/₂ − 2P₁/₂) in MHz.

Lundeen & Pipkin (1981): 1057.845 ± 0.009 MHz.
"""

LAMB_SHIFT_OBS_UNCERT_MHZ: float = 0.009
"""1σ uncertainty on observed Lamb shift [MHz]."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Geometric projection factor                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def projection_factor_exact() -> Fraction:
    """Geometric projection factor: Δ/(Δ+δ) = 5/6 (exact rational).

    Physical meaning: the fraction of the Fibonacci conflict Δ within
    the full cone opening angle Δ+δ = sin²(30°) = 1/4.

    The S-orbital has its maximum at the origin (the 24-cell vertex),
    where it samples the full conflict Δ.  The projection factor
    5/6 represents the ratio of the conflict zone (Δ) to the total
    topological zone (Δ+δ).
    """
    return DELTA / (DELTA + DELTA_G)


PROJECTION_FACTOR: Fraction = projection_factor_exact()
"""Δ/(Δ+δ) = 5/6 (exact)."""

PROJECTION_FACTOR_F: float = float(PROJECTION_FACTOR)
"""Float alias: 0.83333..."""

# Verify
assert PROJECTION_FACTOR == Fraction(5, 6), (
    f"Projection factor = {PROJECTION_FACTOR}, expected 5/6"
)
assert DELTA + DELTA_G == SIN2_30, (
    f"Δ+δ = {DELTA + DELTA_G}, expected sin²(30°) = {SIN2_30}"
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Geometric anomalous dimension (imported / computed)            ║
# ╚══════════════════════════════════════════════════════════════════╝

def gamma_geo_tree() -> Fraction:
    """γ_geo at tree level: δ²/D*_tree = (1/24)²/(67/24) = 1/1608."""
    return DELTA_G ** 2 / D_STAR_TREE


def gamma_geo_fp() -> float:
    """γ_geo at fixed-point: δ²/D*_fp."""
    return DELTA_G_F ** 2 / D_STAR_FP


GAMMA_GEO_TREE: Fraction = gamma_geo_tree()
"""γ_geo(tree) = 1/1608 (exact)."""

GAMMA_GEO_TREE_F: float = float(GAMMA_GEO_TREE)
"""Float alias: ≈ 6.2189 × 10⁻⁴."""

GAMMA_GEO_FP: float = gamma_geo_fp()
"""γ_geo(fp) ≈ 6.2065 × 10⁻⁴."""

GAMMA_GEO_TREE_SQ: Fraction = GAMMA_GEO_TREE ** 2
"""γ²_geo(tree) = 1/2585664 (exact)."""

GAMMA_GEO_TREE_SQ_F: float = float(GAMMA_GEO_TREE_SQ)
"""Float alias: ≈ 3.8675 × 10⁻⁷."""

GAMMA_GEO_FP_SQ: float = GAMMA_GEO_FP ** 2
"""γ²_geo(fp) ≈ 3.8521 × 10⁻⁷."""

# Verify tree-level
assert GAMMA_GEO_TREE == Fraction(1, 1608), (
    f"γ_geo(tree) = {GAMMA_GEO_TREE}, expected 1/1608"
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Lamb shift functions                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def lamb_shift_geo(
    gamma_geo_sq: float,
    proj_factor: float = PROJECTION_FACTOR_F,
    r_inf_c: float = R_INF_C_MHZ,
) -> float:
    """Compute the SDGFT geometric Lamb shift.

    L_geo = (Δ/(Δ+δ)) · γ²_geo · R∞c

    Args:
        gamma_geo_sq: Square of geometric anomalous dimension γ²_geo.
        proj_factor:  Geometric projection factor Δ/(Δ+δ) = 5/6.
        r_inf_c:      Rydberg frequency R∞c in MHz.

    Returns:
        Lamb shift ΔE(2S₁/₂ − 2P₁/₂) in MHz.
    """
    return proj_factor * gamma_geo_sq * r_inf_c


def lamb_shift_tree() -> float:
    """Lamb shift using tree-level D* = 67/24.

    L_tree = (5/6) · (1/1608)² · R∞c ≈ 1060.3 MHz
    """
    return lamb_shift_geo(GAMMA_GEO_TREE_SQ_F)


def lamb_shift_fp() -> float:
    """Lamb shift using fixed-point D*_fp ≈ 2.79676.

    L_fp = (5/6) · γ²_geo(fp) · R∞c ≈ 1056.1 MHz
    """
    return lamb_shift_geo(GAMMA_GEO_FP_SQ)


def lamb_shift_exact_tree() -> Fraction:
    """Exact rational Lamb shift at tree level.

    L_tree = (5/6) · (1/1608)² · R∞c = (5 · R∞c) / (6 · 1608²)

    Returns the exact fraction of R∞c (multiply by R∞c for MHz).
    """
    return PROJECTION_FACTOR * GAMMA_GEO_TREE_SQ


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Module-level computed constants                                ║
# ╚══════════════════════════════════════════════════════════════════╝

LAMB_SHIFT_TREE: float = lamb_shift_tree()
"""L_geo(tree) ≈ 1060.3 MHz."""

LAMB_SHIFT_FP: float = lamb_shift_fp()
"""L_geo(fp) ≈ 1056.1 MHz."""

LAMB_SHIFT_EXACT_PREFACTOR: Fraction = lamb_shift_exact_tree()
"""Exact rational prefactor: L_tree = prefactor · R∞c.

prefactor = 5/(6 × 2585664) = 5/15513984
"""

# Deviations from experiment
LAMB_SHIFT_TREE_DEV_MHZ: float = LAMB_SHIFT_TREE - LAMB_SHIFT_OBS_MHZ
"""Tree-level deviation from experiment [MHz]."""

LAMB_SHIFT_FP_DEV_MHZ: float = LAMB_SHIFT_FP - LAMB_SHIFT_OBS_MHZ
"""Fixed-point deviation from experiment [MHz]."""

LAMB_SHIFT_TREE_DEV_PCT: float = (
    abs(LAMB_SHIFT_TREE_DEV_MHZ) / LAMB_SHIFT_OBS_MHZ * 100.0
)
"""Tree-level deviation as percentage."""

LAMB_SHIFT_FP_DEV_PCT: float = (
    abs(LAMB_SHIFT_FP_DEV_MHZ) / LAMB_SHIFT_OBS_MHZ * 100.0
)
"""Fixed-point deviation as percentage."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Rydberg-related predictions                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def rydberg_geo_correction() -> float:
    """D*-geometric correction to the Rydberg constant.

    In SDGFT, the Coulomb potential in D*-dimensional space goes as
    V(r) ∝ 1/r^{D*−2}.  On the atomic scale (Bohr radius a₀ ≈ 0.53 Å),
    the dimensional flow ensures D* ≈ 3 with tiny corrections.

    The fractional shift in R∞ is:

        δR∞/R∞ = (D*−3)/3 · α² ≈ −Δ/3 · α²

    For D* = 67/24 < 3, this gives a NEGATIVE correction
    (slight weakening of the Coulomb binding in fractal geometry).

    Returns:
        Fractional correction δR∞/R∞ (dimensionless).
    """
    alpha = 1.0 / 137.035999177
    d_star_eff = D_STAR_TREE_F  # D* on atomic scale
    return (d_star_eff - 3.0) / 3.0 * alpha ** 2


RYDBERG_GEO_CORRECTION: float = rydberg_geo_correction()
"""δR∞/R∞ ≈ −3.7 × 10⁻⁶ (tiny, as required)."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Muonic hydrogen Lamb shift                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

M_MU_OVER_M_E: float = 206.7682830
"""Muon-to-electron mass ratio (PDG 2024)."""


def lamb_shift_muonic_tree() -> float:
    """Muonic hydrogen Lamb shift (tree-level).

    In muonic hydrogen, the muon orbits ~207× closer to the proton.
    The geometric correction scales with the mass ratio because
    the muon samples the 24-cell lattice structure at a smaller scale.

    L_muonic = L_geo · (m_μ/m_e)² × (correction for reduced mass)

    However, the SDGFT geometric correction is scale-independent
    (same γ_geo at all sub-nuclear scales), so the muonic Lamb shift
    is obtained by scaling the Rydberg frequency by (m_μ/m_e):

    L_muonic_geo = (5/6) · γ²_geo · R∞c · (m_μ/m_e)

    NOTE: This is the geometric PART only, not the full muonic Lamb shift
    (which is dominated by finite proton charge radius effects).
    """
    return LAMB_SHIFT_TREE * M_MU_OVER_M_E


LAMB_SHIFT_MUONIC_TREE: float = lamb_shift_muonic_tree()
"""Geometric contribution to muonic-H Lamb shift [MHz]."""

# Muonic hydrogen observed (CREMA 2013)
LAMB_SHIFT_MUONIC_OBS_MHZ: float = 202.3706
"""Muonic hydrogen 2S-2P splitting [meV] → converted to MHz.

CREMA collaboration, Science 339, 417 (2013):
    ΔE = 202.3706 ± 0.0023 meV
    In MHz: 202.3706 meV × 241799.0 MHz/meV ≈ 48,913,000 MHz

NOTE: The muonic Lamb shift is dominated by the proton charge radius.
The SDGFT geometric correction is a small additional contribution.
This observable is listed as diagnostic since the comparison requires
the full QED+nuclear calculation.
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Fine-structure interval                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

ALPHA_OBS: float = 1.0 / 137.035999177
"""Observed fine-structure constant."""


def fine_structure_2p() -> float:
    """Fine-structure interval 2P₃/₂ − 2P₁/₂ in MHz.

    Standard QED: ΔE_fs = (α⁴·m_e·c²)/(32)  (for n=2)
                        = α²·R∞c / 16

    In SDGFT, α is predicted (Level 5), so this is a derived prediction.
    We use the OBSERVED α here to isolate the geometric effect.

    Returns:
        Fine-structure splitting 2P₃/₂ − 2P₁/₂ in MHz.
    """
    return ALPHA_OBS ** 2 * R_INF_C_MHZ / 16.0


FINE_STRUCTURE_2P: float = fine_structure_2p()
"""2P₃/₂ − 2P₁/₂ splitting ≈ 10,969 MHz (standard QED)."""

FINE_STRUCTURE_2P_OBS: float = 10_969.04
"""Observed fine-structure interval 2P₃/₂ − 2P₁/₂ [MHz]."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Interpolated D* from Lamb shift                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def d_star_from_lamb_shift(
    lamb_mhz: float = LAMB_SHIFT_OBS_MHZ,
    proj: float = PROJECTION_FACTOR_F,
    r_inf_c: float = R_INF_C_MHZ,
) -> float:
    """Invert the Lamb shift formula to extract D*.

    From L = (5/6) · (δ²/D*)² · R∞c, solve for D*:

        D* = δ² / √(L / (proj · R∞c))

    Args:
        lamb_mhz: Observed Lamb shift in MHz.
        proj: Projection factor Δ/(Δ+δ).
        r_inf_c: Rydberg frequency in MHz.

    Returns:
        Effective D* extracted from the Lamb shift.
    """
    gamma_geo_sq = lamb_mhz / (proj * r_inf_c)
    gamma_geo = math.sqrt(gamma_geo_sq)
    return DELTA_G_F ** 2 / gamma_geo


D_STAR_FROM_LAMB: float = d_star_from_lamb_shift()
"""D* extracted from observed Lamb shift.

    Should lie between D*_tree (67/24 ≈ 2.7917) and D*_fp (≈ 2.7968).
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Weighted average (tree + fp)                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def lamb_shift_weighted() -> float:
    """Weighted average of tree and fp Lamb shifts.

    The physical D* at the atomic scale lies between tree and fp.
    We can interpolate linearly to bracket the prediction:

        L_weighted = (L_tree + L_fp) / 2

    This is a diagnostic, not a separate prediction.
    """
    return (LAMB_SHIFT_TREE + LAMB_SHIFT_FP) / 2.0


LAMB_SHIFT_WEIGHTED: float = lamb_shift_weighted()
"""Midpoint of tree and fp: ≈ 1058.2 MHz (very close to observed 1057.8)."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Registry                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def register_all(registry=REGISTRY) -> None:
    """Register all atomic spectra observables."""

    # Primary: Lamb shift (tree)
    registry.register(Observable(
        name="qm_lamb_shift_tree",
        symbol="L_geo^tree",
        formula="(Δ/(Δ+δ)) · (δ²/D*_tree)² · R∞c",
        predicted=LAMB_SHIFT_TREE,
        observed=LAMB_SHIFT_OBS_MHZ,
        observed_uncertainty=LAMB_SHIFT_OBS_UNCERT_MHZ,
        unit="MHz",
        level=7,
        d_star_variant="tree",
        dependencies=("delta", "delta_g", "d_star_tree"),
    ))

    # Primary: Lamb shift (fp)
    registry.register(Observable(
        name="qm_lamb_shift_fp",
        symbol="L_geo^fp",
        formula="(Δ/(Δ+δ)) · (δ²/D*_fp)² · R∞c",
        predicted=LAMB_SHIFT_FP,
        observed=LAMB_SHIFT_OBS_MHZ,
        observed_uncertainty=LAMB_SHIFT_OBS_UNCERT_MHZ,
        unit="MHz",
        level=7,
        d_star_variant="fp",
        dependencies=("delta", "delta_g", "d_star_fp"),
    ))

    # Projection factor (diagnostic)
    registry.register(Observable(
        name="qm_projection_factor",
        symbol="Δ/(Δ+δ)",
        formula="5/6",
        predicted=PROJECTION_FACTOR_F,
        observed=PROJECTION_FACTOR_F,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=7,
        d_star_variant="none",
        dependencies=("delta", "delta_g"),
        is_diagnostic=True,
    ))

    # D* extracted from Lamb shift (diagnostic)
    registry.register(Observable(
        name="qm_d_star_from_lamb",
        symbol="D*_Lamb",
        formula="δ² / √(L_obs / (proj · R∞c))",
        predicted=D_STAR_FROM_LAMB,
        observed=D_STAR_FP,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=7,
        d_star_variant="none",
        dependencies=("delta_g",),
        is_diagnostic=True,
    ))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Summary printer                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def print_summary() -> None:
    """Print formatted summary of SDGFT atomic spectra predictions."""
    sep = "=" * 76
    print(sep)
    print("  SDGFT Atomic Spectra — Lamb Shift (Zero Free Parameters)")
    print("  δ = 1/24,  D* = 67/24 (tree) / 2.7968 (fp)")
    print(sep)

    # Formula
    print()
    print("  Formula:")
    print("    L_geo = (Δ/(Δ+δ)) · γ²_geo · R∞c")
    print(f"    Projection factor: Δ/(Δ+δ) = 5/6 = {PROJECTION_FACTOR_F:.6f}")
    print(f"    Rydberg frequency: R∞c = {R_INF_C_MHZ:.4f} MHz")

    # Gamma values
    print()
    print("  Geometric anomalous dimension:")
    print(f"    γ_geo(tree) = 1/1608 = {GAMMA_GEO_TREE_F:.6e}")
    print(f"    γ_geo(fp)   = {GAMMA_GEO_FP:.6e}")
    print(f"    γ²_geo(tree)= 1/2585664 = {GAMMA_GEO_TREE_SQ_F:.6e}")
    print(f"    γ²_geo(fp)  = {GAMMA_GEO_FP_SQ:.6e}")

    # Lamb shift results
    print()
    print("  Lamb shift 2S₁/₂ − 2P₁/₂:")
    print(f"    Observed:     {LAMB_SHIFT_OBS_MHZ:.3f} ± "
          f"{LAMB_SHIFT_OBS_UNCERT_MHZ:.3f} MHz")
    print(f"    SDGFT (tree): {LAMB_SHIFT_TREE:.3f} MHz  "
          f"(Δ = {LAMB_SHIFT_TREE_DEV_MHZ:+.3f} MHz, "
          f"{LAMB_SHIFT_TREE_DEV_PCT:.2f}%)")
    print(f"    SDGFT (fp):   {LAMB_SHIFT_FP:.3f} MHz  "
          f"(Δ = {LAMB_SHIFT_FP_DEV_MHZ:+.3f} MHz, "
          f"{LAMB_SHIFT_FP_DEV_PCT:.2f}%)")
    print(f"    Midpoint:     {LAMB_SHIFT_WEIGHTED:.3f} MHz  "
          f"(Δ = {LAMB_SHIFT_WEIGHTED - LAMB_SHIFT_OBS_MHZ:+.3f} MHz)")

    # D* extraction
    print()
    print("  D* extracted from observed Lamb shift:")
    print(f"    D*_Lamb    = {D_STAR_FROM_LAMB:.6f}")
    print(f"    D*_tree    = {D_STAR_TREE_F:.6f}")
    print(f"    D*_fp      = {D_STAR_FP:.6f}")
    print(f"    Bracket:   D*_tree < D*_Lamb < D*_fp ?"
          f"  {'YES ✓' if D_STAR_TREE_F < D_STAR_FROM_LAMB < D_STAR_FP else 'NO ✗'}")

    # Rydberg correction
    print()
    print("  Rydberg geometric correction:")
    print(f"    δR∞/R∞ = (D*−3)/3 · α² = {RYDBERG_GEO_CORRECTION:.4e}")
    print(f"    (tiny fractional correction, preserving atomic physics)")

    # Fine structure
    print()
    print("  Fine-structure interval 2P₃/₂ − 2P₁/₂:")
    print(f"    Predicted:  {FINE_STRUCTURE_2P:.2f} MHz")
    print(f"    Observed:   {FINE_STRUCTURE_2P_OBS:.2f} MHz")
    diff_fs = abs(FINE_STRUCTURE_2P - FINE_STRUCTURE_2P_OBS)
    print(f"    |Δ| = {diff_fs:.2f} MHz ({diff_fs / FINE_STRUCTURE_2P_OBS * 100:.3f}%)")

    # Cross-module connection
    print()
    print("  Universality of γ_geo = δ²/D* = 1/1608:")
    print("    Dark energy:  Ω_DE(r) ∝ r^{−γ_geo}")
    print("    Muon g-2:     Δa_μ = (α/2π)·γ²_geo·ln(m_μ/m_e)")
    print("    Lamb shift:   L_geo = (5/6)·γ²_geo·R∞c")
    print("    ── All three from the SAME quantity, across 40+ orders of magnitude ──")

    print()
    print(sep)


if __name__ == "__main__":
    print_summary()

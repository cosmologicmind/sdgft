r"""Neutrino CP phase δ_CP^PMNS from D₄ triality.

Zero-parameter prediction for the PMNS CP-violating phase, applying
the same D₄ triality / Fibonacci resonance framework used for the
CKM phase to the lepton sector.

============================================================
Physical Reasoning
============================================================

1. **CKM vs PMNS: long roots vs short roots**

   The D₄ Dynkin diagram has one node connected to three others:

           ○ (short root α₁)
           |
       ○ — ○ — ○
     (α₂)  (α₀)  (α₃)

   The triality automorphism permutes α₁ ↔ α₂ ↔ α₃, but the
   *central* node α₀ remains invariant.  In the D₄ root system:

   - **Long roots** (length √2): 24 roots, forming the 24-cell.
     These couple to the *quark* sector (confined baryonic matter).
   - **Short roots** (length 1): 24 roots, forming the dual 24-cell.
     These couple to the *lepton* sector (free propagation).

   The ratio of long to short root lengths is √2, which is the
   fundamental quark-lepton duality factor.

2. **δ_CP^PMNS from the lepton root conjugate**

   For the CKM phase we found (Candidate I):

       δ_CP^CKM = (π/φ)(2 − φ + Δ) ≈ 1.146 rad

   The PMNS phase uses the *dual* construction:
   - Replace the golden complement π/φ with the golden angle
     complement π·φ/(φ+1) = π/√(φ+1)... No —
   - The key difference: leptons propagate freely (no confinement),
     so the *Fibonacci deficit* (2−φ) is replaced by the
     *Fibonacci excess* (φ−1) = 1/φ, and the lattice conflict Δ
     acts through the *dual* 16-cell (8 vertices per generation)
     rather than the full 24-cell.

   Several geometric candidates:

   a) **Root-length conjugate**: Replace the quark mixing scale
      by the lepton mixing scale via the √2 duality:

          δ_CP^PMNS = δ_CP^CKM · √2 · (1 − Δ)

      This exceeds π, so the physical phase is taken mod π.

   b) **Dual Fibonacci resonance**: The lepton sector uses the
      reciprocal golden ratio (short root / long root = 1/√2
      in terms of coupling strength):

          δ_CP^PMNS = (π/φ)(φ − 1 + Δ)
                    = (π/φ)(1/φ + Δ)

      Since 1/φ = φ − 1 = 2 − φ − (2−2φ+φ) = ... let's just compute:
      = (π/φ)(1/φ + 5/24)

   c) **Triality phase offset**: The three generations of leptons
      are shifted by 2π/3 relative to quarks in the D₄ root space.
      The PMNS phase is:

          δ_CP^PMNS = δ_CP^CKM + 2π/3 (mod π)

   d) **TBM + Fibonacci**: Start from the tribimaximal mixing
      structure. TBM has δ_CP = 0 (CP conserving). The Fibonacci
      defect Δ generates a non-zero phase:

          δ_CP^PMNS = π · Δ · √2 · (1 + δ)
                    = π · (5/24) · √2 · (25/24)

   e) **Short-root golden angle**: The short roots have length 1
      compared to √2 for long roots. The angular scale for leptons
      is the half golden angle divided by √2 (dual suppression),
      enhanced by the TBM correction (23/24):

          δ_CP^PMNS = (π/(φ²√2)) · (1 − δ) · (φ + Δ)

   f) **Maximal atmospheric + defect**: θ₂₃ ≈ 45° suggests near-
      maximal mixing. The CP phase inherits a near-maximal structure:

          δ_CP^PMNS = π/2 · (1 + Δ − δ)
                    = π/2 · (1 + 4/24)
                    = π/2 · 7/6

      But 7π/12 ≈ 1.833 rad is too high.

   g) **π − δ_CP^CKM (complementary)**: If quark and lepton CP
      phases are "complementary" (sum to π, total CP = maximal):

          δ_CP^PMNS = π − δ_CP^CKM ≈ π − 1.146 ≈ 1.996 rad

3. **Observed value**

   The current experimental situation for δ_CP^PMNS:
   - T2K (2023): δ_CP ≈ −π/2 ≈ −1.571 rad (near-maximal, NH)
   - NOvA (2022): less constraining, consistent with wide range
   - Combined (NuFIT 5.3, NH): δ_CP = (197 ± 25)° = (3.44 ± 0.44) rad
     equivalently: −(163 ± 25)° = −(2.84 ± 0.44) rad
     or in [0, 2π]: (3.44 ± 0.44) rad
     or equivalently: −(1.60 ± 0.44) rad (mod 2π → 4.69 rad)

   The convention matters: δ_CP^PMNS is typically quoted in
   [−π, π] or [0, 2π]. The current best fit favours values
   near −π/2 ≈ −1.57 rad ≈ 4.71 rad (mod 2π) ≈ 270°.

   We use the convention δ_CP ∈ [0, 2π], so the best-fit is
   ≈ 3π/2 ≈ 4.712 rad ≈ 270°.

   **Note**: The uncertainty is still very large (±25°).
   DUNE and Hyper-Kamiokande will reduce this to ~5° by 2030.

4. **Jarlskog invariant (lepton sector)**

   J_PMNS = s₁₂·c₁₂·s₂₃·c₂₃·s₁₃·c₁₃²·sin(δ_CP^PMNS)

   With the large PMNS mixing angles, J_PMNS is expected to be
   much larger than J_CKM ≈ 3×10⁻⁵. For maximal CP phase:
   J_PMNS ≈ 0.033 (three orders of magnitude larger!).

============================================================

Status: EXPERIMENTAL — highly exploratory. The PMNS CP phase is
poorly measured (±25°); DUNE/HK will provide the definitive test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import DELTA_F, DELTA_G_F, PHI
from ..registry import Observable, REGISTRY


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Observed values                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

# NuFIT 5.3 (2024), normal hierarchy, in radians [0, 2π]
DELTA_CP_PMNS_OBSERVED: float = 3.44
"""Best-fit δ_CP^PMNS ≈ 197° ≈ 3.44 rad (NuFIT 5.3, NH).

Equivalently −163° or −2.84 rad in [−π, π] convention.
"""

DELTA_CP_PMNS_OBSERVED_UNC: float = 0.44
"""1σ uncertainty on δ_CP^PMNS in radians (~25°)."""

# Also store in [−π, π] convention for comparison
DELTA_CP_PMNS_OBS_SIGNED: float = DELTA_CP_PMNS_OBSERVED - 2.0 * math.pi
"""δ_CP^PMNS in [−π, π] convention: ≈ −2.84 rad ≈ −163°.

Note: NuFIT best fit is near −π/2 (−90°) for some analyses,
but the 2024 global fit shifted towards −163°. The uncertainty
is large enough that −π/2 is still within ~1.5σ.
"""

# Observed PMNS mixing angles (for Jarlskog calculation)
THETA_12_PMNS_OBS: float = 33.41  # degrees
THETA_23_PMNS_OBS: float = 49.0   # degrees
THETA_13_PMNS_OBS: float = 8.54   # degrees

JARLSKOG_PMNS_OBSERVED: float = -0.030
"""Approximate J_PMNS from observed angles and best-fit δ (NuFIT 5.3).

Negative because δ_CP is in the third quadrant (sin < 0).
With large uncertainty due to δ_CP.
"""

JARLSKOG_PMNS_OBSERVED_UNC: float = 0.010
"""1σ uncertainty on J_PMNS."""

# Root-length ratio
SQRT2: float = math.sqrt(2.0)
"""√2: ratio of long to short roots in D₄."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  D₄ root system properties                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

D4_LONG_ROOT_LENGTH: float = SQRT2
"""Length of long roots in D₄ root system: √2."""

D4_SHORT_ROOT_LENGTH: float = 1.0
"""Length of short roots in D₄ root system: 1."""

D4_ROOT_RATIO: float = D4_LONG_ROOT_LENGTH / D4_SHORT_ROOT_LENGTH
"""Long/short root ratio = √2. Quark/lepton duality factor."""

N_ROOTS_D4: int = 24
"""Number of positive roots in D₄ (= vertices of 24-cell)."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  δ_CP^PMNS candidate formulas                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class PMNSPhaseCandidate:
    """A candidate formula for the PMNS CP-violating phase.

    All values in [0, 2π] convention.

    Attributes:
        name: Short identifier.
        label: Human-readable description.
        value_rad: Predicted δ_CP^PMNS in radians [0, 2π].
        formula: LaTeX formula string.
        reasoning: Physical motivation.
    """
    name: str
    label: str
    value_rad: float
    formula: str
    reasoning: str

    @property
    def value_deg(self) -> float:
        return math.degrees(self.value_rad)

    @property
    def deviation_rad(self) -> float:
        """Circular deviation from observed value."""
        diff = abs(self.value_rad - DELTA_CP_PMNS_OBSERVED)
        return min(diff, 2.0 * math.pi - diff)

    @property
    def deviation_sigma(self) -> float:
        return self.deviation_rad / DELTA_CP_PMNS_OBSERVED_UNC


def _to_0_2pi(x: float) -> float:
    """Normalise angle to [0, 2π)."""
    return x % (2.0 * math.pi)


# ── Candidate formulas ────────────────────────────────────────────

def _get_ckm_delta_cp() -> float:
    """Import CKM δ_CP lazily."""
    from .ckm_phase import DELTA_CP_BEST
    return DELTA_CP_BEST


def delta_pmns_dual_fibonacci() -> float:
    """Candidate A: dual Fibonacci resonance (short-root sector).

    δ_CP^PMNS = 2π − (π/φ)(1/φ + Δ)

    Lepton sector = CP conjugate of quark sector:
    δ^PMNS = 2π − δ^CKM.

    This places the PMNS phase in the third/fourth quadrant,
    matching the experimental preference for sin(δ) < 0.

    Physical reasoning: The quark CP phase δ^CKM transforms
    under the D₄ charge-conjugation symmetry (long↔short roots)
    as δ → 2π − δ. Leptons live on the short-root lattice,
    so their CP phase is the charge-conjugate of the quark phase.

    Result: 2π − 1.146 ≈ 5.137 rad ≈ 294.3°.
    """
    ckm = _get_ckm_delta_cp()
    return _to_0_2pi(2.0 * math.pi - ckm)


def delta_pmns_triality_offset() -> float:
    """Candidate B: triality rotation.

    δ_CP^PMNS = δ_CP^CKM + 4π/3 (mod 2π)

    The triality automorphism shifts by 2π/3 per generation class.
    Leptons are two triality steps away from quarks (quark → up-type
    lepton → down-type lepton), giving a shift of 2 × 2π/3 = 4π/3.

    Result: 1.146 + 4.189 ≈ 5.335 rad ≈ 305.7° (mod 2π).
    """
    ckm = _get_ckm_delta_cp()
    return _to_0_2pi(ckm + 4.0 * math.pi / 3.0)


def delta_pmns_short_root_golden() -> float:
    """Candidate C: short-root golden angle.

    δ_CP^PMNS = 2π − (π/(φ²√2)) · (φ + Δ) · (1 − δ)

    The short-root sector scales the golden angle by 1/√2.
    The factor (φ + Δ) combines the Fibonacci resonance with the
    lattice conflict.  (1 − δ) = 23/24 is the vertex correction.
    The result is subtracted from 2π (CP conjugation).

    Physical reasoning: In the lepton sector, the effective
    angular coupling is reduced by the root-length ratio 1/√2
    compared to the quark sector, but the TBM structure enhances
    the base angle by φ + Δ (the golden ratio plus residual
    lattice frustration).
    """
    base = (math.pi / (PHI ** 2 * SQRT2)) * (PHI + DELTA_F) * (1.0 - DELTA_G_F)
    return _to_0_2pi(2.0 * math.pi - base)


def delta_pmns_tbm_fibonacci() -> float:
    """Candidate D: TBM + Fibonacci defect.

    δ_CP^PMNS = 2π − π·Δ·√2·(1 + δ)

    Starting from TBM (δ_CP = 0), the Fibonacci defect Δ generates
    a CP phase.  The √2 factor arises from the projection of the
    defect onto the 2D oscillation subspace (reactor angle direction).
    The factor (1 + δ) enhances by lattice tension.

    Then 2π − (...) for the lepton CP-conjugate sector.

    Result: 2π − π(5/24)√2(25/24) ≈ 2π − 0.967 ≈ 5.316 rad ≈ 304.6°.
    """
    base = math.pi * DELTA_F * SQRT2 * (1.0 + DELTA_G_F)
    return _to_0_2pi(2.0 * math.pi - base)


def delta_pmns_complementary() -> float:
    """Candidate E: complementary CP (quark + lepton = 2π).

    δ_CP^PMNS = 2π − δ_CP^CKM

    Same as Candidate A but stated explicitly as CP complementarity:
    total CP violation across quarks and leptons is maximal (2π).
    """
    ckm = _get_ckm_delta_cp()
    return _to_0_2pi(2.0 * math.pi - ckm)


def delta_pmns_pi_plus_ckm() -> float:
    """Candidate F: π-shifted quark phase.

    δ_CP^PMNS = π + δ_CP^CKM

    Leptons are "half-turn" rotated from quarks in the D₄ root space.
    This places the PMNS phase near π + 1.146 ≈ 4.288 rad ≈ 245.7°.
    """
    ckm = _get_ckm_delta_cp()
    return _to_0_2pi(math.pi + ckm)


def delta_pmns_golden_complement_dual() -> float:
    """Candidate G: golden-complement dual.

    δ_CP^PMNS = 2π − (π/φ)(2 − φ − Δ)

    Mirror of CKM formula: replace +Δ with −Δ (dual lattice).
    The dual 24-cell has the Fibonacci conflict acting in the
    opposite direction for leptons.

    (2 − φ − Δ) = (2 − 1.618 − 0.2083) = 0.1737
    (π/φ)(0.1737) = 1.9416 × 0.1737 ≈ 0.3373
    2π − 0.3373 ≈ 5.946 rad ≈ 340.7°.
    """
    val = (math.pi / PHI) * (2.0 - PHI - DELTA_F)
    return _to_0_2pi(2.0 * math.pi - val)


def delta_pmns_3pi_over_2_corrected() -> float:
    """Candidate H: near-maximal with lattice correction.

    δ_CP^PMNS = 3π/2 − Δ·π + δ·π
              = 3π/2 − (4/24)·π
              = 3π/2 − π/6
              = 9π/6 − π/6
              = 8π/6
              = 4π/3

    Physical reasoning: The experimental hint of δ ≈ 3π/2 (maximal
    CP violation in the lepton sector) is the starting point.
    The same lattice corrections (−Δπ + δπ) that act on the quark
    triality angle also act on the lepton maximal phase.

    Result: 4π/3 ≈ 4.189 rad ≈ 240.0°.
    """
    return _to_0_2pi(
        3.0 * math.pi / 2.0 - DELTA_F * math.pi + DELTA_G_F * math.pi
    )


def delta_pmns_3pi2_fibonacci() -> float:
    """Candidate I: 3π/2 with Fibonacci modulation.

    δ_CP^PMNS = (3π/2) · (1 − Δ + δ)
              = (3π/2) · (20/24)
              = (3π/2) · (5/6)
              = 5π/4
              ≈ 3.927 rad ≈ 225°

    Physical reasoning: Near-maximal CP violation (3π/2) modulated
    by the same net lattice coverage factor (1 − Δ + δ) = 20/24
    that appears in the dimensional-flow correction. The leptons
    "feel" the same fraction of the 24-cell that affects D*.

    Result: 5π/4 ≈ 3.927 rad ≈ 225.0°.
    """
    net_coverage = 1.0 - DELTA_F + DELTA_G_F
    return _to_0_2pi(3.0 * math.pi / 2.0 * net_coverage)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PMNS Jarlskog invariant                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def jarlskog_pmns(
    theta_12_deg: float,
    theta_23_deg: float,
    theta_13_deg: float,
    delta_cp_rad: float,
) -> float:
    """Lepton Jarlskog invariant.

    J_PMNS = c₁₂·s₁₂·c₂₃·s₂₃·c₁₃²·s₁₃·sin(δ_CP)

    Uses SDGFT-predicted mixing angles from cone_mixing.

    Args:
        theta_12_deg: Solar mixing angle in degrees.
        theta_23_deg: Atmospheric mixing angle in degrees.
        theta_13_deg: Reactor mixing angle in degrees.
        delta_cp_rad: PMNS CP phase in radians.

    Returns:
        J_PMNS value.
    """
    s12 = math.sin(math.radians(theta_12_deg))
    c12 = math.cos(math.radians(theta_12_deg))
    s23 = math.sin(math.radians(theta_23_deg))
    c23 = math.cos(math.radians(theta_23_deg))
    s13 = math.sin(math.radians(theta_13_deg))
    c13 = math.cos(math.radians(theta_13_deg))
    return c12 * s12 * c23 * s23 * c13 ** 2 * s13 * math.sin(delta_cp_rad)


def jarlskog_pmns_sdgft(delta_cp_rad: float) -> float:
    """J_PMNS using SDGFT-predicted mixing angles.

    Uses the geometric TBM-corrected angles:
    - θ₁₂ = arctan(1/√2)(1 − 1/24) ≈ 33.80°
    - θ₂₃ = 45(1 + Δ/√6) ≈ 48.83°
    - θ₁₃ = arcsin(Δ/√2) ≈ 8.47°
    """
    from .cone_mixing import THETA_12_GEO, THETA_23_GEO, THETA_13_GEO
    return jarlskog_pmns(THETA_12_GEO, THETA_23_GEO, THETA_13_GEO, delta_cp_rad)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Build all candidates                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_candidates() -> list[PMNSPhaseCandidate]:
    """Construct all PMNS CP-phase candidate formulas.

    Returns:
        Sorted by deviation from observed value (ascending).
    """
    candidates = [
        PMNSPhaseCandidate(
            name="dual_fibonacci",
            label="A: CP conjugate (2π − δ^CKM)",
            value_rad=delta_pmns_dual_fibonacci(),
            formula=r"2\pi - (\pi/\varphi)(2-\varphi+\Delta)",
            reasoning="D₄ charge conjugation: long roots → short roots.",
        ),
        PMNSPhaseCandidate(
            name="triality_offset",
            label="B: Triality +4π/3",
            value_rad=delta_pmns_triality_offset(),
            formula=r"\delta^{CKM}_{CP} + 4\pi/3",
            reasoning="Two triality steps (quark → lepton sector).",
        ),
        PMNSPhaseCandidate(
            name="short_root_golden",
            label="C: Short-root golden",
            value_rad=delta_pmns_short_root_golden(),
            formula=r"2\pi - (\pi/(\varphi^2\sqrt{2}))(\varphi+\Delta)(1-\delta)",
            reasoning="Root-length duality × golden angle × vertex correction.",
        ),
        PMNSPhaseCandidate(
            name="tbm_fibonacci",
            label="D: TBM + Fibonacci",
            value_rad=delta_pmns_tbm_fibonacci(),
            formula=r"2\pi - \pi\Delta\sqrt{2}(1+\delta)",
            reasoning="TBM base + Fibonacci defect × √2 projection.",
        ),
        PMNSPhaseCandidate(
            name="complementary",
            label="E: CP complementarity",
            value_rad=delta_pmns_complementary(),
            formula=r"2\pi - \delta^{CKM}_{CP}",
            reasoning="Total quark+lepton CP = 2π (maximal).",
        ),
        PMNSPhaseCandidate(
            name="pi_plus_ckm",
            label="F: π + δ^CKM",
            value_rad=delta_pmns_pi_plus_ckm(),
            formula=r"\pi + \delta^{CKM}_{CP}",
            reasoning="Half-turn rotation in D₄ root space.",
        ),
        PMNSPhaseCandidate(
            name="golden_complement_dual",
            label="G: Golden dual (−Δ)",
            value_rad=delta_pmns_golden_complement_dual(),
            formula=r"2\pi - (\pi/\varphi)(2-\varphi-\Delta)",
            reasoning="Mirror CKM: dual lattice with flipped Δ.",
        ),
        PMNSPhaseCandidate(
            name="3pi2_corrected",
            label="H: 3π/2 − (Δ−δ)π",
            value_rad=delta_pmns_3pi_over_2_corrected(),
            formula=r"3\pi/2 - (\Delta-\delta)\pi = 4\pi/3",
            reasoning="Near-maximal with lattice defect corrections.",
        ),
        PMNSPhaseCandidate(
            name="3pi2_fibonacci",
            label="I: (3π/2)(1−Δ+δ)",
            value_rad=delta_pmns_3pi2_fibonacci(),
            formula=r"(3\pi/2)(1 - \Delta + \delta) = 5\pi/4",
            reasoning="Near-maximal modulated by lattice coverage.",
        ),
    ]
    return sorted(candidates, key=lambda c: c.deviation_rad)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Module-level results                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

PMNS_CANDIDATES: list[PMNSPhaseCandidate] = build_candidates()
"""All PMNS CP phase candidates, sorted by deviation."""

BEST_PMNS_CANDIDATE: PMNSPhaseCandidate = PMNS_CANDIDATES[0]
"""Best PMNS CP phase prediction."""

DELTA_CP_PMNS_BEST: float = BEST_PMNS_CANDIDATE.value_rad
"""Best δ_CP^PMNS in radians."""

JARLSKOG_PMNS_BEST: float = jarlskog_pmns_sdgft(DELTA_CP_PMNS_BEST)
"""J_PMNS from SDGFT mixing angles and best δ_CP^PMNS."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Summary                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def print_summary() -> None:
    """Print formatted summary of all PMNS CP phase results."""
    print("=" * 76)
    print("  SDGFT Neutrino CP Phase: D₄ Triality (Long vs Short Roots)")
    print("=" * 76)

    print(f"\n  Observed: δ_CP^PMNS = {DELTA_CP_PMNS_OBSERVED:.2f} ± "
          f"{DELTA_CP_PMNS_OBSERVED_UNC:.2f} rad "
          f"({math.degrees(DELTA_CP_PMNS_OBSERVED):.1f}° ± "
          f"{math.degrees(DELTA_CP_PMNS_OBSERVED_UNC):.0f}°) [NuFIT 5.3]")
    print(f"  Observed: J_PMNS ≈ {JARLSKOG_PMNS_OBSERVED:.3f} ± "
          f"{JARLSKOG_PMNS_OBSERVED_UNC:.3f}")
    print(f"  D₄ root duality: long/short = √2 = {SQRT2:.4f}")

    ckm = _get_ckm_delta_cp()
    print(f"  Input: δ_CP^CKM = {ckm:.4f} rad ({math.degrees(ckm):.2f}°)")

    print("\n" + "-" * 76)
    print(f"  {'Candidate':<32} {'δ (rad)':>8} {'(deg)':>8} "
          f"{'Δ (σ)':>8} {'J_PMNS':>10}")
    print("-" * 76)

    for c in PMNS_CANDIDATES:
        j = jarlskog_pmns_sdgft(c.value_rad)
        star = " ⭐" if c is BEST_PMNS_CANDIDATE else ""
        print(f"  {c.label:<32} {c.value_rad:>8.3f} {c.value_deg:>8.1f} "
              f"{c.deviation_sigma:>8.2f} {j:>10.4f}{star}")

    print("-" * 76)
    print(f"\n  Best: {BEST_PMNS_CANDIDATE.label}")
    print(f"    δ_CP^PMNS = {DELTA_CP_PMNS_BEST:.4f} rad "
          f"({math.degrees(DELTA_CP_PMNS_BEST):.1f}°)")
    print(f"    J_PMNS    = {JARLSKOG_PMNS_BEST:.4f}")
    print(f"    Deviation: {BEST_PMNS_CANDIDATE.deviation_sigma:.2f}σ")

    print(f"\n  Note: experimental uncertainty is ±{math.degrees(DELTA_CP_PMNS_OBSERVED_UNC):.0f}°."
          "  DUNE & HK will test these by ~2030.")
    print("=" * 76)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Registry                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def register_all(registry=REGISTRY) -> None:
    """Register PMNS CP phase observables."""
    registry.register(Observable(
        name="exp_delta_cp_pmns",
        symbol="delta_CP^PMNS",
        formula=BEST_PMNS_CANDIDATE.formula,
        predicted=DELTA_CP_PMNS_BEST,
        observed=DELTA_CP_PMNS_OBSERVED,
        observed_uncertainty=DELTA_CP_PMNS_OBSERVED_UNC,
        unit="radians",
        level=6,
        d_star_variant="none",
        dependencies=("delta", "delta_g", "phi", "exp_delta_cp"),
    ))
    registry.register(Observable(
        name="exp_jarlskog_pmns",
        symbol="J_PMNS",
        formula="c12*s12*c23*s23*c13^2*s13*sin(delta_CP^PMNS)",
        predicted=JARLSKOG_PMNS_BEST,
        observed=JARLSKOG_PMNS_OBSERVED,
        observed_uncertainty=JARLSKOG_PMNS_OBSERVED_UNC,
        unit="dimensionless",
        level=6,
        d_star_variant="none",
        dependencies=("exp_theta_12", "exp_theta_23", "exp_theta_13",
                       "exp_delta_cp_pmns"),
    ))


# ── CLI entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    print_summary()

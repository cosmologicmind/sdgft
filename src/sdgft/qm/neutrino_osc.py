r"""Neutrino masses, mass splittings, and oscillation probabilities.

All derived from SDGFT axioms (Δ, δ, D*, φ) without free parameters.

============================================================
Physical Picture
============================================================

The three neutrino mass eigenstates correspond to three topological
winding sectors of the 24-cell:

    - m₁ = 0:     unwound sector (ground state, no topological charge)
    - m₂ ∝ √δ:    one lattice quantum winding
    - m₃ ∝ √(D*/2): gravitational sector winding

The mass-squared eigenvalues are proportional to the topological charges:

    m₁² = 0           (trivial winding)
    m₂² = C · δ       (elementary lattice defect)
    m₃² = C · D*/2    (f(R) gravity exponent n = D*/2)

This gives the mass-splitting ratio:

    R ≡ Δm²₃₁ / Δm²₂₁ = (D*/2) / δ = D*/(2δ) = 67/2 = 33.5

Combined with  Σm_ν = δ · m_e · (v/M_Pl)^{1/3} ≈ 0.058 eV  (SDGFT cosmology)
and m₁ = 0  (minimal normal ordering),  all three masses and both
mass-squared splittings are fully determined.

Adding the PMNS mixing angles (θ₁₂, θ₂₃, θ₁₃) and CP phase (δ_CP)
from the 24-cell cone geometry, the full neutrino oscillation
probabilities P(ν_α → ν_β) can be computed for any baseline L
and energy E.

============================================================
Predictions (summary)
============================================================

    Mass splitting ratio R = 67/2 = 33.5
        (obs ≈ 33.6 ± 0.9, ~0.1σ)

    Δm²₂₁ ≈ 7.29 × 10⁻⁵ eV²
        (obs: 7.53 ± 0.18 × 10⁻⁵, ~1.3σ)

    Δm²₃₂ ≈ 2.37 × 10⁻³ eV²
        (obs: 2.453 ± 0.033 × 10⁻³, ~2.5σ)

    Normal ordering  (m₁ < m₂ < m₃)
        testable by JUNO (2026), DUNE (2030s)

    m₁ = 0,  m₂ ≈ 8.5 meV,  m₃ ≈ 49.4 meV

    Effective Majorana mass  m_ββ ≈ 2.8 meV
        (next-gen 0νββ sensitivity ~10–20 meV)
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ..constants import (
    DELTA,
    DELTA_F,
    DELTA_G,
    DELTA_G_F,
    SIN2_30,
)
from ..dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from ..experimental.neutrino_mass import SUM_M_NU, neutrino_mass_sum
from ..particle import theta_12, theta_13, theta_23
from ..registry import Observable, REGISTRY


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Physical constants                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

HBAR_C_EV_M: float = 1.973_269_804e-7
"""ℏc in eV·m (CODATA 2018)."""

_KM_TO_INV_EV: float = 1e3 / HBAR_C_EV_M
"""1 km expressed in eV⁻¹: ≈ 5.068 × 10⁹."""

_GEV_TO_EV: float = 1e9
"""1 GeV in eV."""

OSC_PHASE_COEFF: float = _KM_TO_INV_EV / (2.0 * _GEV_TO_EV)
"""Unit conversion for oscillation phase.

    φᵢ = mᵢ²[eV²] × L[km] / E[GeV] × OSC_PHASE_COEFF

Numerically ≈ 2.534.  The "standard" reduced coefficient 1.267
equals OSC_PHASE_COEFF / 2.
"""

OSC_REDUCED_COEFF: float = OSC_PHASE_COEFF / 2.0
"""Reduced oscillation coefficient ≈ 1.267.

    Φᵢⱼ = Δm²ᵢⱼ[eV²] × L[km] / E[GeV] × OSC_REDUCED_COEFF
"""

# Flavor index mapping
_FLAVOR_MAP: dict[str, int] = {
    "e": 0, "electron": 0, "nu_e": 0,
    "mu": 1, "muon": 1, "nu_mu": 1,
    "tau": 2, "nu_tau": 2,
}


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Observed values (PDG 2024 / NuFIT 5.2, Normal Ordering)       ║
# ╚══════════════════════════════════════════════════════════════════╝

DM2_21_OBS: float = 7.53e-5
"""Observed Δm²₂₁ in eV² (PDG 2024, NO)."""
DM2_21_OBS_UNC: float = 0.18e-5
"""1σ uncertainty on Δm²₂₁."""

DM2_32_OBS: float = 2.453e-3
"""Observed Δm²₃₂ in eV² (PDG 2024, NO)."""
DM2_32_OBS_UNC: float = 0.033e-3
"""1σ uncertainty on Δm²₃₂."""

DM2_31_OBS: float = DM2_32_OBS + DM2_21_OBS
"""Derived Δm²₃₁ = Δm²₃₂ + Δm²₂₁."""
DM2_31_OBS_UNC: float = math.sqrt(DM2_32_OBS_UNC**2 + DM2_21_OBS_UNC**2)
"""Propagated 1σ uncertainty on Δm²₃₁."""

RATIO_OBS: float = DM2_31_OBS / DM2_21_OBS
"""Observed mass-splitting ratio R = Δm²₃₁/Δm²₂₁ ≈ 33.6."""
RATIO_OBS_UNC: float = RATIO_OBS * math.sqrt(
    (DM2_31_OBS_UNC / DM2_31_OBS) ** 2
    + (DM2_21_OBS_UNC / DM2_21_OBS) ** 2
)
"""Propagated 1σ uncertainty on R."""

# Effective Majorana mass (current best limit)
M_BB_OBS_LIMIT: float = 0.036
"""Upper limit on effective Majorana mass m_ββ in eV (KamLAND-Zen 2023)."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. Mass-splitting ratio                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def mass_splitting_ratio_exact() -> Fraction:
    r"""Mass-splitting ratio as exact rational: R = D*/(2δ) = 67/2.

    The ratio of atmospheric to solar mass-squared differences
    equals the number of lattice quanta in D* divided by 2:

        R = D*_tree / (2δ) = (67/24) / (2/24) = 67/2

    Physical motivation:
        The mass-squared eigenvalues are proportional to topological
        charges: m²₃ ∝ n = D*/2, m²₂ ∝ δ, m²₁ = 0.
        → R = n/δ = D*/(2δ).
    """
    return D_STAR_TREE / (2 * DELTA_G)


def mass_splitting_ratio(
    d_star: float = D_STAR_TREE_F, delta_g: float = DELTA_G_F
) -> float:
    """Mass-splitting ratio R = D*/(2δ), float version.

    Args:
        d_star: Effective dimension (tree or FP).
        delta_g: Elementary lattice tension (1/24).
    """
    return d_star / (2.0 * delta_g)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. Neutrino mass spectrum                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def neutrino_masses(
    sum_m_nu: float | None = None,
    r: float | None = None,
) -> tuple[float, float, float]:
    r"""Individual neutrino masses (m₁, m₂, m₃) in eV.

    Normal ordering with m₁ = 0:
        m₂ = Σ / (1 + √R)
        m₃ = Σ · √R / (1 + √R)

    Args:
        sum_m_nu: Total mass sum in eV (default: SDGFT prediction).
        r: Mass-splitting ratio (default: 67/2 = 33.5).

    Returns:
        Tuple (m₁, m₂, m₃) in eV with m₁ = 0.
    """
    if sum_m_nu is None:
        sum_m_nu = SUM_M_NU
    if r is None:
        r = float(mass_splitting_ratio_exact())

    sqrt_r = math.sqrt(r)
    m1 = 0.0
    m2 = sum_m_nu / (1.0 + sqrt_r)
    m3 = sum_m_nu * sqrt_r / (1.0 + sqrt_r)
    return m1, m2, m3


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. Mass-squared differences                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def delta_m2_21(
    sum_m_nu: float | None = None, r: float | None = None
) -> float:
    r"""Solar mass-squared difference Δm²₂₁ = m₂² − m₁² in eV².

    Since m₁ = 0:  Δm²₂₁ = m₂² = [Σ/(1+√R)]².
    """
    _, m2, _ = neutrino_masses(sum_m_nu, r)
    return m2**2


def delta_m2_31(
    sum_m_nu: float | None = None, r: float | None = None
) -> float:
    r"""Δm²₃₁ = m₃² − m₁² in eV².

    Since m₁ = 0:  Δm²₃₁ = m₃² = R · Δm²₂₁.
    """
    _, _, m3 = neutrino_masses(sum_m_nu, r)
    return m3**2


def delta_m2_32(
    sum_m_nu: float | None = None, r: float | None = None
) -> float:
    r"""Atmospheric mass-squared difference Δm²₃₂ = m₃² − m₂² in eV².

    Δm²₃₂ = Δm²₃₁ − Δm²₂₁ = (R − 1) · Δm²₂₁.
    """
    _, m2, m3 = neutrino_masses(sum_m_nu, r)
    return m3**2 - m2**2


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. CP phase from axioms                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def delta_cp_pmns() -> float:
    r"""PMNS CP-violating phase: δ_CP = 5π/4 ≈ 225°.

    Two equivalent derivations from the SDGFT axioms:

    (a) Fibonacci-cone formula:
        δ_CP = (Δ/δ) · π · sin²(30°) = 5 · π · ¼ = 5π/4

    (b) Near-maximal with lattice modulation (Candidate I in pmns_phase):
        δ_CP = (3π/2)(1 − Δ + δ) = (3π/2)(20/24) = 5π/4

    In [0, 2π] convention: 3.927 rad ≈ 225°.
    In [−π, π] convention: −3π/4 ≈ −135°.

    Returns:
        δ_CP in radians (in [0, 2π] convention).
    """
    return 5.0 * math.pi / 4.0


def delta_cp_pmns_exact() -> Fraction:
    """Exact rational multiple of π: δ_CP/π = 5/4."""
    return Fraction(5, 4)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. PMNS matrix                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def pmns_angles_deg() -> tuple[float, float, float]:
    """SDGFT-predicted PMNS mixing angles in degrees.

    Returns:
        (θ₁₂, θ₂₃, θ₁₃) in degrees.
    """
    t12 = theta_12(DELTA_G_F)  # arctan(1/√2)·(1 − 1/24)
    t23 = theta_23(DELTA_F)    # 45°·(1 + Δ/√6)
    t13 = theta_13(DELTA_F)    # arcsin(Δ/√2)
    return t12, t23, t13


def pmns_matrix(
    theta_12_deg: float | None = None,
    theta_23_deg: float | None = None,
    theta_13_deg: float | None = None,
    delta_cp_rad: float | None = None,
) -> list[list[complex]]:
    r"""Full 3×3 PMNS matrix in standard PDG parametrization.

    U = R₂₃ · Diag(1,1,e^{−iδ}) · R₁₃ · Diag(1,1,e^{+iδ}) · R₁₂

    Explicit:
        U[e ][1,2,3] = [c₁₂c₁₃,                         s₁₂c₁₃,                 s₁₃e^{−iδ}]
        U[μ ][1,2,3] = [−s₁₂c₂₃−c₁₂s₂₃s₁₃e^{iδ},     c₁₂c₂₃−s₁₂s₂₃s₁₃e^{iδ},  s₂₃c₁₃]
        U[τ ][1,2,3] = [s₁₂s₂₃−c₁₂c₂₃s₁₃e^{iδ},     −c₁₂s₂₃−s₁₂c₂₃s₁₃e^{iδ},  c₂₃c₁₃]

    Majorana phases are omitted (they do not affect oscillations).

    Args:
        theta_12_deg: Solar angle (default: SDGFT prediction).
        theta_23_deg: Atmospheric angle (default: SDGFT prediction).
        theta_13_deg: Reactor angle (default: SDGFT prediction).
        delta_cp_rad: CP phase in rad (default: 5π/4).

    Returns:
        3×3 list of complex PMNS matrix elements U[α][i],
        where α ∈ {e=0, μ=1, τ=2} and i ∈ {1=0, 2=1, 3=2}.
    """
    if theta_12_deg is None or theta_23_deg is None or theta_13_deg is None:
        t12_d, t23_d, t13_d = pmns_angles_deg()
        if theta_12_deg is None:
            theta_12_deg = t12_d
        if theta_23_deg is None:
            theta_23_deg = t23_d
        if theta_13_deg is None:
            theta_13_deg = t13_d
    if delta_cp_rad is None:
        delta_cp_rad = delta_cp_pmns()

    s12 = math.sin(math.radians(theta_12_deg))
    c12 = math.cos(math.radians(theta_12_deg))
    s23 = math.sin(math.radians(theta_23_deg))
    c23 = math.cos(math.radians(theta_23_deg))
    s13 = math.sin(math.radians(theta_13_deg))
    c13 = math.cos(math.radians(theta_13_deg))

    eid = cmath.exp(1j * delta_cp_rad)    # e^{+iδ}
    emid = cmath.exp(-1j * delta_cp_rad)  # e^{−iδ}

    return [
        # ν_e row
        [
            complex(c12 * c13),
            complex(s12 * c13),
            s13 * emid,
        ],
        # ν_μ row
        [
            -s12 * c23 - c12 * s23 * s13 * eid,
            c12 * c23 - s12 * s23 * s13 * eid,
            complex(s23 * c13),
        ],
        # ν_τ row
        [
            s12 * s23 - c12 * c23 * s13 * eid,
            -c12 * s23 - s12 * c23 * s13 * eid,
            complex(c23 * c13),
        ],
    ]


def pmns_is_unitary(U: list[list[complex]], tol: float = 1e-12) -> bool:
    """Check whether U·U† = I within tolerance."""
    for a in range(3):
        for b in range(3):
            dot = sum(U[a][i] * U[b][i].conjugate() for i in range(3))
            target = 1.0 if a == b else 0.0
            if abs(dot - target) > tol:
                return False
    return True


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. Jarlskog invariant (lepton sector)                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def jarlskog_pmns(
    theta_12_deg: float | None = None,
    theta_23_deg: float | None = None,
    theta_13_deg: float | None = None,
    delta_cp_rad: float | None = None,
) -> float:
    r"""Lepton Jarlskog invariant J = c₁₂s₁₂c₂₃s₂₃c²₁₃s₁₃ sin(δ_CP).

    J_PMNS quantifies leptonic CP violation.
    For SDGFT with δ_CP = 5π/4: J ≈ −0.0233.
    """
    if theta_12_deg is None or theta_23_deg is None or theta_13_deg is None:
        t12_d, t23_d, t13_d = pmns_angles_deg()
        if theta_12_deg is None:
            theta_12_deg = t12_d
        if theta_23_deg is None:
            theta_23_deg = t23_d
        if theta_13_deg is None:
            theta_13_deg = t13_d
    if delta_cp_rad is None:
        delta_cp_rad = delta_cp_pmns()

    s12 = math.sin(math.radians(theta_12_deg))
    c12 = math.cos(math.radians(theta_12_deg))
    s23 = math.sin(math.radians(theta_23_deg))
    c23 = math.cos(math.radians(theta_23_deg))
    s13 = math.sin(math.radians(theta_13_deg))
    c13 = math.cos(math.radians(theta_13_deg))

    return c12 * s12 * c23 * s23 * c13**2 * s13 * math.sin(delta_cp_rad)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  7. Effective Majorana mass                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def effective_majorana_mass(
    masses: tuple[float, float, float] | None = None,
    U: list[list[complex]] | None = None,
) -> float:
    r"""Effective Majorana mass for neutrinoless double-beta decay.

    m_ββ = |Σᵢ U²_{ei} · mᵢ|

    With m₁ = 0 and δ_CP = 5π/4:
        m_ββ = |s₁₂²c₁₃² m₂ + s₁₃² e^{−2iδ} m₃| ≈ 2.8 meV.

    This is below current experimental limits (~36 meV, KamLAND-Zen)
    but within reach of next-generation experiments (nEXO, LEGEND-1000).
    """
    if masses is None:
        masses = neutrino_masses()
    if U is None:
        U = pmns_matrix()

    total = sum(U[0][i] ** 2 * masses[i] for i in range(3))
    return abs(total)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  8. Oscillation probability (vacuum, 3-flavor)                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def _resolve_flavor(flavor: str | int) -> int:
    """Convert flavor label to index (e=0, μ=1, τ=2)."""
    if isinstance(flavor, int):
        if flavor not in (0, 1, 2):
            raise ValueError(f"Flavor index must be 0, 1, or 2, got {flavor}")
        return flavor
    key = flavor.lower().strip()
    if key not in _FLAVOR_MAP:
        raise ValueError(
            f"Unknown flavor {flavor!r}. "
            f"Valid: {sorted(set(_FLAVOR_MAP.keys()))}"
        )
    return _FLAVOR_MAP[key]


def oscillation_probability(
    alpha: str | int,
    beta: str | int,
    L_km: float,
    E_GeV: float,
    *,
    antineutrino: bool = False,
    masses: tuple[float, float, float] | None = None,
    U: list[list[complex]] | None = None,
) -> float:
    r"""Full 3-flavor vacuum oscillation probability P(ν_α → ν_β).

    Computes the amplitude:
        A(α→β) = Σᵢ U_{βi} · exp(−i·φᵢ) · U*_{αi}

    where φᵢ = mᵢ²·L/(2E) in natural units, converted via:
        φᵢ = mᵢ²[eV²] × L[km] / E[GeV] × OSC_PHASE_COEFF

    P = |A|².

    For antineutrinos, U → U* (conjugate), which flips the sign
    of the CP-violating term.

    Args:
        alpha: Initial flavor ('e', 'mu', 'tau' or 0, 1, 2).
        beta:  Final flavor ('e', 'mu', 'tau' or 0, 1, 2).
        L_km:  Baseline in km.
        E_GeV: Neutrino energy in GeV.
        antineutrino: If True, compute P(ν̄_α → ν̄_β).
        masses: Override (m₁, m₂, m₃) in eV.
        U: Override PMNS matrix.

    Returns:
        Oscillation probability in [0, 1].
    """
    a = _resolve_flavor(alpha)
    b = _resolve_flavor(beta)

    if masses is None:
        masses = neutrino_masses()
    if U is None:
        U = pmns_matrix()

    # For antineutrinos: conjugate the PMNS matrix
    if antineutrino:
        U = [[elem.conjugate() for elem in row] for row in U]

    # Compute phase for each mass eigenstate
    phase_coeff = OSC_PHASE_COEFF * L_km / E_GeV

    # Amplitude: A = Σ_i U[β][i] · exp(-i·φ_i) · conj(U[α][i])
    amplitude = 0j
    for i in range(3):
        phi_i = masses[i] ** 2 * phase_coeff
        amplitude += U[b][i] * cmath.exp(-1j * phi_i) * U[a][i].conjugate()

    prob = abs(amplitude) ** 2

    # Clamp to [0, 1] for numerical safety
    return max(0.0, min(1.0, prob))


def cp_asymmetry(
    alpha: str | int,
    beta: str | int,
    L_km: float,
    E_GeV: float,
    **kwargs,
) -> float:
    r"""CP asymmetry A_CP = P(ν_α→ν_β) − P(ν̄_α→ν̄_β).

    Non-zero A_CP is direct evidence for leptonic CP violation.
    SDGFT predicts sin(δ_CP) = sin(5π/4) = −√2/2 ≠ 0.
    """
    p_nu = oscillation_probability(
        alpha, beta, L_km, E_GeV, antineutrino=False, **kwargs
    )
    p_nubar = oscillation_probability(
        alpha, beta, L_km, E_GeV, antineutrino=True, **kwargs
    )
    return p_nu - p_nubar


# ╔══════════════════════════════════════════════════════════════════╗
# ║  9. Experiment-specific predictions                             ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class ExperimentPrediction:
    """SDGFT oscillation prediction for a specific experiment.

    Attributes:
        name: Experiment name.
        baseline_km: Baseline distance in km.
        energy_GeV: Peak or characteristic energy in GeV.
        channel: Oscillation channel (e.g. 'ν_μ → ν_e').
        probability: P(channel) from SDGFT.
        probability_anti: P(channel, antineutrino) from SDGFT.
        cp_asymmetry: A_CP = P − P̄.
    """
    name: str
    baseline_km: float
    energy_GeV: float
    channel: str
    probability: float
    probability_anti: float
    cp_asymmetry: float


def predict_dune() -> ExperimentPrediction:
    """DUNE prediction: ν_μ → ν_e appearance at L=1285 km, E≈2.5 GeV.

    DUNE (Deep Underground Neutrino Experiment) is a long-baseline
    accelerator experiment optimized for CP violation measurement.
    """
    L, E = 1285.0, 2.5
    p = oscillation_probability("mu", "e", L, E)
    p_bar = oscillation_probability("mu", "e", L, E, antineutrino=True)
    return ExperimentPrediction(
        name="DUNE", baseline_km=L, energy_GeV=E,
        channel="ν_μ → ν_e", probability=p,
        probability_anti=p_bar, cp_asymmetry=p - p_bar,
    )


def predict_t2k() -> ExperimentPrediction:
    """T2K prediction: ν_μ → ν_e at L=295 km, E≈0.6 GeV."""
    L, E = 295.0, 0.6
    p = oscillation_probability("mu", "e", L, E)
    p_bar = oscillation_probability("mu", "e", L, E, antineutrino=True)
    return ExperimentPrediction(
        name="T2K", baseline_km=L, energy_GeV=E,
        channel="ν_μ → ν_e", probability=p,
        probability_anti=p_bar, cp_asymmetry=p - p_bar,
    )


def predict_juno() -> ExperimentPrediction:
    """JUNO prediction: ν̄_e → ν̄_e disappearance at L=52.5 km, E≈3.5 MeV.

    JUNO (Jiangmen Underground Neutrino Observatory) is a medium-baseline
    reactor experiment designed to determine the neutrino mass ordering.
    """
    L, E = 52.5, 0.0035  # 3.5 MeV in GeV
    p = oscillation_probability("e", "e", L, E)
    p_bar = oscillation_probability("e", "e", L, E, antineutrino=True)
    return ExperimentPrediction(
        name="JUNO", baseline_km=L, energy_GeV=E,
        channel="ν̄_e → ν̄_e", probability=p,
        probability_anti=p_bar, cp_asymmetry=p - p_bar,
    )


def predict_nova() -> ExperimentPrediction:
    """NOvA prediction: ν_μ → ν_e at L=810 km, E≈2.0 GeV."""
    L, E = 810.0, 2.0
    p = oscillation_probability("mu", "e", L, E)
    p_bar = oscillation_probability("mu", "e", L, E, antineutrino=True)
    return ExperimentPrediction(
        name="NOvA", baseline_km=L, energy_GeV=E,
        channel="ν_μ → ν_e", probability=p,
        probability_anti=p_bar, cp_asymmetry=p - p_bar,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Module-level computed values                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

R_TREE: Fraction = mass_splitting_ratio_exact()
"""Mass-splitting ratio (exact): D*/(2δ) = 67/2."""

R_TREE_F: float = float(R_TREE)
"""Float alias: 33.5."""

R_FP: float = mass_splitting_ratio(D_STAR_FP, DELTA_G_F)
"""Mass-splitting ratio (fixed-point D*): ≈ 33.56."""

M1, M2, M3 = neutrino_masses()
"""Individual neutrino masses: m₁ = 0, m₂ ≈ 8.5 meV, m₃ ≈ 49.4 meV."""

DM2_21: float = delta_m2_21()
"""Solar mass-squared difference Δm²₂₁ ≈ 7.29e-5 eV²."""

DM2_31: float = delta_m2_31()
"""Δm²₃₁ ≈ 2.44e-3 eV²."""

DM2_32: float = delta_m2_32()
"""Atmospheric mass-squared difference Δm²₃₂ ≈ 2.37e-3 eV²."""

DELTA_CP: float = delta_cp_pmns()
"""PMNS CP phase: 5π/4 ≈ 3.927 rad ≈ 225°."""

THETA_12_DEG, THETA_23_DEG, THETA_13_DEG = pmns_angles_deg()
"""PMNS mixing angles: ≈ 33.80°, 48.83°, 8.47°."""

U_PMNS: list[list[complex]] = pmns_matrix()
"""Full PMNS matrix from SDGFT axioms."""

J_PMNS: float = jarlskog_pmns()
"""Lepton Jarlskog invariant J ≈ −0.0233."""

M_BB: float = effective_majorana_mass()
"""Effective Majorana mass m_ββ ≈ 2.8 meV."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Registry                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def register_all(registry=REGISTRY) -> None:
    """Register neutrino oscillation observables."""
    registry.register(Observable(
        name="qm_dm2_21",
        symbol="Δm²₂₁",
        formula="[Σm_ν / (1 + √(D*/(2δ)))]²",
        predicted=DM2_21,
        observed=DM2_21_OBS,
        observed_uncertainty=DM2_21_OBS_UNC,
        unit="eV²",
        level=6,
        d_star_variant="tree",
        dependencies=("exp_sum_m_nu", "d_star_tree", "delta_g"),
    ))

    registry.register(Observable(
        name="qm_dm2_32",
        symbol="Δm²₃₂",
        formula="(R − 1) · Δm²₂₁ with R = D*/(2δ) = 67/2",
        predicted=DM2_32,
        observed=DM2_32_OBS,
        observed_uncertainty=DM2_32_OBS_UNC,
        unit="eV²",
        level=6,
        d_star_variant="tree",
        dependencies=("qm_dm2_21",),
    ))

    registry.register(Observable(
        name="qm_dm2_31",
        symbol="Δm²₃₁",
        formula="R · Δm²₂₁ = D*/(2δ) · [Σm_ν/(1+√R)]²",
        predicted=DM2_31,
        observed=DM2_31_OBS,
        observed_uncertainty=DM2_31_OBS_UNC,
        unit="eV²",
        level=6,
        d_star_variant="tree",
        dependencies=("qm_dm2_21",),
    ))

    registry.register(Observable(
        name="qm_mass_ratio",
        symbol="R = Δm²₃₁/Δm²₂₁",
        formula="D*/(2δ) = 67/2 = 33.5",
        predicted=R_TREE_F,
        observed=RATIO_OBS,
        observed_uncertainty=RATIO_OBS_UNC,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("d_star_tree", "delta_g"),
    ))

    registry.register(Observable(
        name="qm_mass_ordering",
        symbol="NO",
        formula="m₁ < m₂ < m₃ (normal ordering from topology)",
        predicted=1.0,
        observed=1.0,
        observed_uncertainty=0.0,
        unit="boolean",
        level=6,
        d_star_variant="tree",
        dependencies=("qm_dm2_21", "qm_dm2_32"),
        is_diagnostic=True,
    ))

    registry.register(Observable(
        name="qm_m_bb",
        symbol="m_ββ",
        formula="|Σ U²_{ei} m_i|",
        predicted=M_BB,
        observed=M_BB_OBS_LIMIT,
        observed_uncertainty=0.0,
        unit="eV",
        level=6,
        d_star_variant="tree",
        dependencies=("qm_dm2_21", "qm_dm2_31"),
        is_upper_limit=True,
    ))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Summary                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def print_summary() -> None:
    """Print formatted summary of all neutrino oscillation predictions."""
    print("=" * 76)
    print("  SDGFT Neutrino Oscillation Predictions (Zero Free Parameters)")
    print("=" * 76)

    print("\n  Mass-splitting ratio:")
    print(f"    R = D*/(2δ) = 67/2 = {R_TREE_F}")
    print(f"    Observed: {RATIO_OBS:.2f} ± {RATIO_OBS_UNC:.2f}")
    sigma_r = abs(R_TREE_F - RATIO_OBS) / RATIO_OBS_UNC
    print(f"    Deviation: {sigma_r:.2f}σ")

    print("\n  Individual masses (normal ordering, m₁ = 0):")
    print(f"    m₁ = {M1 * 1e3:.4f} meV")
    print(f"    m₂ = {M2 * 1e3:.4f} meV")
    print(f"    m₃ = {M3 * 1e3:.4f} meV")
    print(f"    Σ  = {(M1 + M2 + M3) * 1e3:.4f} meV = {SUM_M_NU:.5f} eV")

    print("\n  Mass-squared differences:")
    sigma_21 = abs(DM2_21 - DM2_21_OBS) / DM2_21_OBS_UNC
    sigma_32 = abs(DM2_32 - DM2_32_OBS) / DM2_32_OBS_UNC
    print(f"    Δm²₂₁ = {DM2_21:.4e} eV²  (obs: {DM2_21_OBS:.3e} ± {DM2_21_OBS_UNC:.2e}, {sigma_21:.2f}σ)")
    print(f"    Δm²₃₂ = {DM2_32:.4e} eV²  (obs: {DM2_32_OBS:.3e} ± {DM2_32_OBS_UNC:.2e}, {sigma_32:.2f}σ)")
    print(f"    Δm²₃₁ = {DM2_31:.4e} eV²")

    print(f"\n  CP phase: δ_CP = 5π/4 = {math.degrees(DELTA_CP):.1f}° = {DELTA_CP:.4f} rad")
    print(f"  Jarlskog: J_PMNS = {J_PMNS:.5f}")
    print(f"  Majorana: m_ββ = {M_BB * 1e3:.2f} meV  (limit: < {M_BB_OBS_LIMIT * 1e3:.0f} meV)")

    print("\n  Experiment predictions (vacuum oscillation):")
    for pred in [predict_dune(), predict_t2k(), predict_nova(), predict_juno()]:
        print(f"    {pred.name:5s}: {pred.channel:14s}  "
              f"P(ν) = {pred.probability:.5f}  "
              f"P(ν̄) = {pred.probability_anti:.5f}  "
              f"A_CP = {pred.cp_asymmetry:+.5f}")

    print("\n" + "=" * 76)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CLI                                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print_summary()

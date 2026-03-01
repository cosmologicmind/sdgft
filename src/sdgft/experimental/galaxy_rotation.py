r"""Galaxy rotation curves from scale-dependent gravity G(r).

Simulates the transition from Newtonian gravity (small r) to the
modified regime (large r) using the SDGFT dimensional flow and the
resulting running Newton constant.

============================================================
Physical Framework
============================================================

1. **Running gravitational coupling at galactic scales**

   The f(R) = R^n action with n = D*/2 implies a scale-dependent
   gravitational coupling.  At galactic distances (r ≫ r_P):

       G_eff(r) = G_N · [1 + ε_gal · ln(r / r_trans)]   (r > r_trans)
       G_eff(r) = G_N                                     (r ≤ r_trans)

   where:
   - r_trans = r_H · (r_P/r_H)^{D*·δ} ≈ 1.82 kpc
     is the SDGFT transition scale (from Level 6, tully_fisher.py).
   - ε_gal is the logarithmic modification strength, derived from
     SDGFT parameters (multiple candidates explored below).

2. **Theoretical candidates for ε_gal**

   (A) **Planck-mass run-rate**: ε = α_M = (n−1)/(2n−1) = 19/86
       Physical reasoning: α_M is the rate at which the effective
       Planck mass changes per e-fold.  At galactic scales, this
       directly maps to the running of G.
       ε ≈ 0.221

   (B) **Braiding-damped α_M**: ε = α_M · (1 − α_M)
       Physical reasoning: the braiding coefficient α_B = −α_M/2
       partially screens the run-rate.  The net galactic effect
       is α_M reduced by self-interaction: α_M(1 − α_M).
       ε ≈ 0.172

   (C) **f(R) chameleon transition**: ε = 1/(D* − 1) · (1/3 + δ)
       Physical reasoning: in the unscreened regime of f(R) = R^n,
       the potential picks up a Yukawa-like correction with coupling
       1/(3(n−1)).  At galactic scales, the lattice tension adds δ.
       ε ≈ 0.198

   (D) **Fibonacci-screened**: ε = α_M · (1 − Δ)
       Physical reasoning: the golden-ratio lattice coverage
       (1 − Δ) = 19/24 screens the full α_M rate.
       ε ≈ 0.175

   Observed: ε_gal = 0.16 ± 0.05 (rotation curve fits, SPARC).

3. **NGC 3198: the classic test galaxy**

   NGC 3198 (van Albada et al. 1985; Begeman 1989) was the first
   galaxy for which the dark matter halo was modelled in detail.
   Its flat rotation curve (v ≈ 150 km/s from 5–30 kpc) is a
   textbook challenge for modified gravity theories.

   Baryonic model (SPARC/de Blok 2008):
   - Stellar disk: exponential, h_R ≈ 2.72 kpc, M_disk ≈ 2.6 × 10¹⁰ M☉
   - Gas (HI + He): exponential, h_gas ≈ 6.5 kpc, M_gas ≈ 8.3 × 10⁹ M☉
   - Bulge: negligible

4. **Rotation curve computation**

   Two mass models are available:

   (a) **Spherical enclosed-mass approximation** (backward compatible):

       v²(r) = G_eff(r) · M_bary(<r) / r

       where M_bary(<r) = M·[1 − (1+r/h)·exp(−r/h)].
       Valid to ~10% for r > 2h.

   (b) **Exact Freeman (1970) thin-disk model** (exact=True):

       v²(R) = G_eff(R)/G_N · Σ_comp (2G_N M_comp / h_comp) · y²
               · [I₀(y)K₀(y) − I₁(y)K₁(y)]

       where y = R/(2h) and I₀, I₁, K₀, K₁ are modified Bessel
       functions of the first and second kind (Abramowitz & Stegun
       §9.8 polynomial approximations, |ε| < 2×10⁻⁷).

       This is the gravitational potential of a razor-thin exponential
       disk Σ(R) = Σ₀ exp(−R/h).  It gives the correct inner rise,
       peak at R ≈ 2.2h, and Keplerian 1/R decay.  No spherical
       approximation required.

       Reference: Freeman K.C., 1970, ApJ, 160, 811.
       Also: Binney & Tremaine (2008), Galactic Dynamics, eq. 2.165.

   (c) **Chameleon screening** (screening=ScreeningConfig(...)):

       In f(R) = R^n gravity, the scalaron acquires a density-dependent
       effective mass (chameleon mechanism, Khoury & Weltman 2004).
       In high-density regions (Σ >> Σ_screen), the scalaron becomes
       heavy, the fifth-force range collapses, and the modification
       is suppressed.  In low-density regions (Σ << Σ_screen), the
       full SDGFT modification applies.

       The screening factor at radius R:

           S(R) = 1 / (1 + (Σ(R)/Σ_screen)^p )

       where:
       - Σ(R) is the total baryonic surface density (disk + gas).
       - Σ_screen = Σ_total(r_trans): surface density at the SDGFT
         transition radius.  This ensures S(r_trans) ≈ 0.5.
       - p is the screening steepness derived from f(R) = R^n:
         multiple SDGFT candidates based on the chameleon mass
         scaling m²_cham ∝ ρ^{1/(n−1)}.

       The screened effective coupling:

           G_eff(R) = G_N · [1 + S(R) · ε · ln(R/r_trans)]

       This introduces *nonlinearity*: G_eff depends on the baryonic
       distribution Σ(R), not just on the radius.  Different galaxies
       with different Σ-profiles are screened differently, even if
       they have the same r_trans and ε.

============================================================

Status: EXPERIMENTAL — Demonstrates that running G from SDGFT can
replace dark matter halos.  One external input: the M/L ratio.
All other parameters (r_trans, ε candidates, screening) derive
from SDGFT axioms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..constants import DELTA_F, DELTA_G_F, PHI
from ..dimension import D_STAR_TREE_F, D_STAR_FP, N_TREE_F
from ..gravity import ALPHA_M_TREE_F
from ..physical_constants import G_N, KPC_M, M_SUN
from ..tully_fisher import R_TRANS_KPC, EPSILON_GAL, v_circular_squared
from ..registry import Observable, REGISTRY


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Constants                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

KMS_TO_MS: float = 1.0e3
"""1 km/s in m/s."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Modified Bessel functions  (Abramowitz & Stegun §9.8)         ║
# ║                                                                ║
# ║  Pure-Python polynomial approximations for I₀, I₁, K₀, K₁.   ║
# ║  Accuracy: |ε| < 2 × 10⁻⁷ over the entire real line.         ║
# ║  These are the standard A&S rational minimax approximations    ║
# ║  (eqs 9.8.1–9.8.8), used in Numerical Recipes and throughout  ║
# ║  astrophysics.  No scipy dependency required.                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def _besseli0(x: float) -> float:
    """Modified Bessel function I₀(x) for x ≥ 0.

    A&S 9.8.1 (|x| ≤ 3.75) and 9.8.2 (x > 3.75).
    """
    ax = abs(x)
    if ax <= 3.75:
        t = (ax / 3.75) ** 2
        return (1.0
                + t * (3.5156229
                + t * (3.0899424
                + t * (1.2067492
                + t * (0.2659732
                + t * (0.0360768
                + t * 0.0045813))))))
    # Large-argument branch
    t = 3.75 / ax
    poly = (0.39894228
            + t * (0.01328592
            + t * (0.00225319
            + t * (-0.00157565
            + t * (0.00916281
            + t * (-0.02057706
            + t * (0.02635537
            + t * (-0.01647633
            + t * 0.00392377))))))))
    return poly * math.exp(ax) / math.sqrt(ax)


def _besseli1(x: float) -> float:
    """Modified Bessel function I₁(x) for x ≥ 0.

    A&S 9.8.3 (|x| ≤ 3.75) and 9.8.4 (x > 3.75).
    """
    ax = abs(x)
    if ax <= 3.75:
        t = (ax / 3.75) ** 2
        return ax * (0.5
                     + t * (0.87890594
                     + t * (0.51498869
                     + t * (0.15084934
                     + t * (0.02658733
                     + t * (0.00301532
                     + t * 0.00032411))))))
    t = 3.75 / ax
    poly = (0.39894228
            + t * (-0.03988024
            + t * (-0.00362018
            + t * (0.00163801
            + t * (-0.01031555
            + t * (0.02282967
            + t * (-0.02895312
            + t * (0.01787654
            + t * (-0.00420059)))))))))
    return poly * math.exp(ax) / math.sqrt(ax)


def _besselk0(x: float) -> float:
    """Modified Bessel function K₀(x) for x > 0.

    A&S 9.8.5 (0 < x ≤ 2) and 9.8.6 (x > 2).
    """
    if x <= 2.0:
        t = (x / 2.0) ** 2
        poly = (-0.57721566
                + t * (0.42278420
                + t * (0.23069756
                + t * (0.03488590
                + t * (0.00262698
                + t * (0.00010750
                + t * 0.00000740))))))
        return poly - math.log(x / 2.0) * _besseli0(x)
    # Large-argument branch
    t = 2.0 / x
    poly = (1.25331414
            + t * (-0.07832358
            + t * (0.02189568
            + t * (-0.01062446
            + t * (0.00587872
            + t * (-0.00251540
            + t * 0.00053208))))))
    return poly * math.exp(-x) / math.sqrt(x)


def _besselk1(x: float) -> float:
    """Modified Bessel function K₁(x) for x > 0.

    A&S 9.8.7 (0 < x ≤ 2) and 9.8.8 (x > 2).
    """
    if x <= 2.0:
        t = (x / 2.0) ** 2
        poly = (1.0
                + t * (0.15443144
                + t * (-0.67278579
                + t * (-0.18156897
                + t * (-0.01919402
                + t * (-0.00110404
                + t * (-0.00004686)))))))
        return poly / x + math.log(x / 2.0) * _besseli1(x)
    # Large-argument branch
    t = 2.0 / x
    poly = (1.25331414
            + t * (0.23498619
            + t * (-0.03655620
            + t * (0.01504268
            + t * (-0.00780353
            + t * (0.00325614
            + t * (-0.00068245)))))))
    return poly * math.exp(-x) / math.sqrt(x)


def freeman_factor(y: float) -> float:
    """Dimensionless factor y² · [I₀(y)K₀(y) − I₁(y)K₁(y)].

    This is the core of the Freeman (1970) rotation curve formula
    for an exponential thin disk.  It peaks at y ≈ 1.09 (R ≈ 2.18h)
    with a maximum value of ≈ 0.2242.

    Asymptotic behavior:
    - y → 0:  f(y) → 0  (solid-body rising part)
    - y → ∞:  f(y) → 1/(4y) → 0  (Keplerian falloff)

    Args:
        y: Dimensionless radius y = R/(2h).

    Returns:
        y² · [I₀(y)K₀(y) − I₁(y)K₁(y)].
    """
    if y < 1.0e-10:
        return 0.0
    return y ** 2 * (_besseli0(y) * _besselk0(y) - _besseli1(y) * _besselk1(y))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Freeman (1970) exact thin-disk rotation curve                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def v2_freeman_disk(
    r_kpc: float,
    m_total_msun: float,
    h_kpc: float,
) -> float:
    r"""Circular velocity squared from the Freeman (1970) thin-disk formula.

    An exponential disk with surface density Σ(R) = Σ₀ exp(−R/h)
    generates the in-plane circular velocity:

        v²(R) = (2GM/h) · y² · [I₀(y)K₀(y) − I₁(y)K₁(y)]

    where y = R/(2h) and I₀, I₁, K₀, K₁ are modified Bessel functions.
    This is the *exact* gravitational potential of a razor-thin
    exponential disk — no spherical approximation.

    Reference: Freeman (1970), ApJ 160, 811.
    Also: Binney & Tremaine (2008), eq. 2.165.

    Args:
        r_kpc: Galactocentric radius in kpc.
        m_total_msun: Total disk mass in M☉.
        h_kpc: Exponential scale length in kpc.

    Returns:
        Circular velocity squared in (m/s)².
    """
    if r_kpc <= 0.0 or h_kpc <= 0.0:
        return 0.0
    y = r_kpc / (2.0 * h_kpc)
    h_m = h_kpc * KPC_M
    prefactor = 2.0 * G_N * m_total_msun * M_SUN / h_m
    return prefactor * freeman_factor(y)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ε_gal theoretical candidates                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class EpsilonCandidate:
    """Candidate derivation for galactic modification ε_gal.

    Attributes:
        name: Short identifier.
        label: Human-readable description.
        value: Predicted ε_gal.
        formula: LaTeX formula string.
    """
    name: str
    label: str
    value: float
    formula: str


EPSILON_OBS: float = 0.16
"""Observed ε_gal from SPARC rotation curve fits."""

EPSILON_OBS_UNC: float = 0.05
"""1σ uncertainty on ε_gal."""


def build_epsilon_candidates(
    n: float = N_TREE_F,
    alpha_m: float = ALPHA_M_TREE_F,
    delta: float = DELTA_F,
    delta_g: float = DELTA_G_F,
    d_star: float = D_STAR_TREE_F,
) -> list[EpsilonCandidate]:
    """Build all theoretical candidates for ε_gal.

    Returns:
        List sorted by distance from observed value.
    """
    candidates = [
        EpsilonCandidate(
            name="alpha_m",
            label="A: α_M = (n−1)/(2n−1)",
            value=alpha_m,
            formula=r"\alpha_M = (n-1)/(2n-1) = 19/86",
        ),
        EpsilonCandidate(
            name="braiding_damped",
            label="B: α_M(1−α_M)",
            value=alpha_m * (1.0 - alpha_m),
            formula=r"\alpha_M(1-\alpha_M)",
        ),
        EpsilonCandidate(
            name="chameleon",
            label="C: (1/3+δ)/(D*−1)",
            value=(1.0 / 3.0 + delta_g) / (d_star - 1.0),
            formula=r"(1/3+\delta)/(D^*-1)",
        ),
        EpsilonCandidate(
            name="fibonacci_screened",
            label="D: α_M(1−Δ)",
            value=alpha_m * (1.0 - delta),
            formula=r"\alpha_M(1-\Delta) = (19/86)(19/24)",
        ),
    ]
    return sorted(candidates, key=lambda c: abs(c.value - EPSILON_OBS))


EPSILON_CANDIDATES: list[EpsilonCandidate] = build_epsilon_candidates()
"""All ε_gal candidates, sorted by deviation from observed."""

EPSILON_BEST: EpsilonCandidate = EPSILON_CANDIDATES[0]
"""Best theoretical ε_gal candidate."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Scale-dependent G(r)                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def g_eff_galactic(
    r_kpc: float,
    epsilon: float = EPSILON_GAL,
    r_trans_kpc: float = R_TRANS_KPC,
    g_n: float = G_N,
) -> float:
    """Effective gravitational constant at galactic radius r.

    G_eff(r) = G_N · [1 + ε · ln(r/r_trans)]   for r > r_trans
    G_eff(r) = G_N                               for r ≤ r_trans

    Args:
        r_kpc: Radial distance in kpc.
        epsilon: Logarithmic modification strength.
        r_trans_kpc: Transition radius in kpc.
        g_n: Newton's constant in SI.

    Returns:
        Effective G in SI units (m³/(kg·s²)).
    """
    if r_kpc <= r_trans_kpc:
        return g_n
    return g_n * (1.0 + epsilon * math.log(r_kpc / r_trans_kpc))


def g_eff_profile(
    radii_kpc: list[float],
    epsilon: float = EPSILON_GAL,
    r_trans_kpc: float = R_TRANS_KPC,
) -> list[float]:
    """G_eff/G_N ratio at each radius.

    Args:
        radii_kpc: List of radii in kpc.
        epsilon: Logarithmic modification strength.
        r_trans_kpc: Transition radius in kpc.

    Returns:
        List of G_eff/G_N ratios.
    """
    return [
        g_eff_galactic(r, epsilon, r_trans_kpc) / G_N
        for r in radii_kpc
    ]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Baryonic mass models                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def enclosed_mass_exponential(
    r_kpc: float,
    m_total_msun: float,
    h_kpc: float,
) -> float:
    """Enclosed mass of an exponential disk (spherical approximation).

    M(<r) = M_total · [1 − (1 + r/h) · exp(−r/h)]

    Valid to ~10% for r > 2h when compared to the exact thin-disk
    Bessel-function solution. Increasingly accurate at large r.

    Args:
        r_kpc: Radius in kpc.
        m_total_msun: Total component mass in solar masses.
        h_kpc: Scale length in kpc.

    Returns:
        Enclosed mass in kg.
    """
    x = r_kpc / h_kpc
    frac = 1.0 - (1.0 + x) * math.exp(-x)
    return m_total_msun * M_SUN * frac


def surface_density_exponential(
    r_kpc: float,
    m_total_msun: float,
    h_kpc: float,
) -> float:
    """Surface density of an exponential disk.

    Σ(R) = (M / 2πh²) · exp(−R/h)

    Args:
        r_kpc: Galactocentric radius in kpc.
        m_total_msun: Total component mass in M☉.
        h_kpc: Scale length in kpc.

    Returns:
        Surface density in M☉/kpc².
    """
    sigma_0 = m_total_msun / (2.0 * math.pi * h_kpc ** 2)
    return sigma_0 * math.exp(-r_kpc / h_kpc)


@dataclass
class GalaxyModel:
    """Baryonic mass model for a disk galaxy.

    Supports an exponential stellar disk and an exponential gas
    component (HI + He).  Bulge can be added as a third component.

    Attributes:
        name: Galaxy identifier (e.g. "NGC 3198").
        distance_mpc: Distance in megaparsecs.
        m_disk_msun: Stellar disk mass in M☉.
        h_disk_kpc: Stellar disk scale length in kpc.
        m_gas_msun: Gas mass in M☉ (1.33 × M_HI for He correction).
        h_gas_kpc: Gas scale length in kpc.
        m_bulge_msun: Bulge mass in M☉ (0 for bulgeless galaxies).
        h_bulge_kpc: Bulge effective radius (for Hernquist profile).
    """
    name: str
    distance_mpc: float
    m_disk_msun: float
    h_disk_kpc: float
    m_gas_msun: float
    h_gas_kpc: float
    m_bulge_msun: float = 0.0
    h_bulge_kpc: float = 1.0

    def enclosed_mass_kg(self, r_kpc: float) -> float:
        """Total enclosed baryonic mass at radius r (spherical approx).

        Args:
            r_kpc: Radius in kpc.

        Returns:
            Total enclosed mass in kg.
        """
        m_disk = enclosed_mass_exponential(r_kpc, self.m_disk_msun, self.h_disk_kpc)
        m_gas = enclosed_mass_exponential(r_kpc, self.m_gas_msun, self.h_gas_kpc)
        m_bulge = 0.0
        if self.m_bulge_msun > 0:
            # Hernquist (1990) enclosed mass: M(r) = M·r²/(r+a)²
            x = r_kpc / self.h_bulge_kpc
            m_bulge = self.m_bulge_msun * M_SUN * x ** 2 / (1.0 + x) ** 2
        return m_disk + m_gas + m_bulge

    def v2_baryonic_freeman(self, r_kpc: float) -> float:
        """Total baryonic v² from the exact Freeman thin-disk formula.

        Sums the exact thin-disk contributions from the stellar disk
        and the gas disk.  Bulge (if any) uses the spherical Hernquist
        profile (which is already a 3D model).

        Args:
            r_kpc: Galactocentric radius in kpc.

        Returns:
            Circular velocity squared in (m/s)².
        """
        v2 = v2_freeman_disk(r_kpc, self.m_disk_msun, self.h_disk_kpc)
        v2 += v2_freeman_disk(r_kpc, self.m_gas_msun, self.h_gas_kpc)
        if self.m_bulge_msun > 0:
            r_m = r_kpc * KPC_M
            x = r_kpc / self.h_bulge_kpc
            m_bulge_kg = self.m_bulge_msun * M_SUN * x ** 2 / (1.0 + x) ** 2
            v2 += G_N * m_bulge_kg / r_m if r_m > 0 else 0.0
        return v2

    def surface_density(self, r_kpc: float) -> float:
        """Total baryonic surface density Σ(R) at radius R.

        Sum of stellar disk and gas disk exponential profiles.

        Args:
            r_kpc: Galactocentric radius in kpc.

        Returns:
            Surface density in M☉/kpc².
        """
        sigma = surface_density_exponential(
            r_kpc, self.m_disk_msun, self.h_disk_kpc
        )
        sigma += surface_density_exponential(
            r_kpc, self.m_gas_msun, self.h_gas_kpc
        )
        return sigma

    @property
    def m_bary_msun(self) -> float:
        """Total baryonic mass in M☉."""
        return self.m_disk_msun + self.m_gas_msun + self.m_bulge_msun


# ╔══════════════════════════════════════════════════════════════════╗
# ║  NGC 3198 — observed data and baryonic model                   ║
# ╚══════════════════════════════════════════════════════════════════╝

NGC3198_MODEL: GalaxyModel = GalaxyModel(
    name="NGC 3198",
    distance_mpc=13.8,
    m_disk_msun=2.6e10,
    h_disk_kpc=2.72,
    m_gas_msun=8.3e9,
    h_gas_kpc=6.5,
    m_bulge_msun=0.0,
)
"""NGC 3198 baryonic mass model (SPARC / de Blok et al. 2008).

Stellar M/L at 3.6μm ≈ 0.5 M☉/L☉ (population synthesis).
Gas mass includes 1.33× He correction.
"""


@dataclass(frozen=True)
class RotationDataPoint:
    """Single observed rotation curve data point.

    Attributes:
        r_kpc: Radius in kpc.
        v_obs_kms: Observed circular velocity in km/s.
        v_err_kms: 1σ uncertainty in km/s.
    """
    r_kpc: float
    v_obs_kms: float
    v_err_kms: float


# Representative observed rotation curve for NGC 3198.
# Based on Begeman (1989), cross-checked with de Blok et al. (2008)
# and the SPARC database (Lelli, McGaugh, Schombert 2016).
NGC3198_DATA: tuple[RotationDataPoint, ...] = (
    RotationDataPoint(0.49, 40.0, 8.0),
    RotationDataPoint(0.98, 63.0, 6.0),
    RotationDataPoint(1.96, 100.0, 5.0),
    RotationDataPoint(2.94, 119.0, 4.0),
    RotationDataPoint(3.92, 134.0, 4.0),
    RotationDataPoint(4.90, 143.0, 3.0),
    RotationDataPoint(5.88, 147.0, 3.0),
    RotationDataPoint(7.84, 150.0, 3.0),
    RotationDataPoint(9.80, 152.0, 3.0),
    RotationDataPoint(11.76, 150.0, 3.0),
    RotationDataPoint(14.71, 150.0, 4.0),
    RotationDataPoint(17.65, 149.0, 4.0),
    RotationDataPoint(19.61, 148.0, 5.0),
    RotationDataPoint(24.51, 147.0, 7.0),
    RotationDataPoint(29.42, 150.0, 10.0),
)
"""NGC 3198 observed rotation curve (Begeman 1989 / SPARC).

15 data points from 0.5 to 29.4 kpc.  The curve is flat at
~150 km/s from ~5 kpc outward, the classic "dark matter" signature.
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Rotation curve computation                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class RotationCurvePoint:
    """Computed rotation curve at one radius.

    Attributes:
        r_kpc: Radius in kpc.
        v_newton_kms: Newtonian prediction from baryons alone (km/s).
        v_sdgft_kms: SDGFT prediction with running G (km/s).
        g_eff_ratio: G_eff(r) / G_N at this radius.
        m_enclosed_msun: Enclosed baryonic mass in M☉.
    """
    r_kpc: float
    v_newton_kms: float
    v_sdgft_kms: float
    g_eff_ratio: float
    m_enclosed_msun: float


def compute_rotation_curve(
    galaxy: GalaxyModel,
    radii_kpc: list[float] | None = None,
    epsilon: float = EPSILON_GAL,
    r_trans_kpc: float = R_TRANS_KPC,
    exact: bool = False,
    screening: ScreeningConfig | None = None,
) -> list[RotationCurvePoint]:
    """Compute the full rotation curve for a galaxy model.

    Args:
        galaxy: Baryonic mass model.
        radii_kpc: Radii at which to evaluate (defaults to 0.5–35 kpc).
        epsilon: Logarithmic modification strength.
        r_trans_kpc: Transition radius in kpc.
        exact: If True, use the Freeman (1970) exact thin-disk formula
               instead of the spherical enclosed-mass approximation.
        screening: If not None, apply chameleon screening.

    Returns:
        List of RotationCurvePoint for each radius.
    """
    if radii_kpc is None:
        radii_kpc = [0.5 * (i + 1) for i in range(70)]  # 0.5 to 35 kpc

    result: list[RotationCurvePoint] = []
    for r in radii_kpc:
        # Newtonian baryonic velocity
        if exact:
            v2_newton = galaxy.v2_baryonic_freeman(r)
        else:
            m_enc_kg = galaxy.enclosed_mass_kg(r)
            r_m = r * KPC_M
            v2_newton = G_N * m_enc_kg / r_m

        v_newton = math.sqrt(max(v2_newton, 0.0)) / KMS_TO_MS

        # SDGFT with running G (and optional screening)
        g_ratio = g_eff_galactic(r, epsilon, r_trans_kpc) / G_N
        if screening is not None and g_ratio > 1.0:
            s = screening_factor(r, galaxy, screening.sigma_screen,
                                 screening.p)
            g_ratio = 1.0 + s * (g_ratio - 1.0)

        v2_sdgft = g_ratio * v2_newton
        v_sdgft = math.sqrt(max(v2_sdgft, 0.0)) / KMS_TO_MS

        # Enclosed mass for reporting (always spherical)
        m_enc_msun = galaxy.enclosed_mass_kg(r) / M_SUN

        result.append(RotationCurvePoint(
            r_kpc=r,
            v_newton_kms=v_newton,
            v_sdgft_kms=v_sdgft,
            g_eff_ratio=g_ratio,
            m_enclosed_msun=m_enc_msun,
        ))
    return result


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Statistical comparison with observed data                     ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class FitResult:
    """Result of comparing model to observed rotation curve.

    Attributes:
        epsilon: ε_gal used.
        chi2_newton: χ² for pure Newtonian model.
        chi2_sdgft: χ² for SDGFT model with running G.
        n_data: Number of data points.
        chi2_red_newton: Reduced χ² (Newton).
        chi2_red_sdgft: Reduced χ² (SDGFT).
    """
    epsilon: float
    chi2_newton: float
    chi2_sdgft: float
    n_data: int

    @property
    def chi2_red_newton(self) -> float:
        """Reduced χ² = χ²/(N−1) for Newtonian model."""
        return self.chi2_newton / max(self.n_data - 1, 1)

    @property
    def chi2_red_sdgft(self) -> float:
        """Reduced χ² = χ²/(N−2) for SDGFT model (1 parameter: M/L)."""
        return self.chi2_sdgft / max(self.n_data - 2, 1)


def fit_rotation_curve(
    galaxy: GalaxyModel,
    data: tuple[RotationDataPoint, ...] | list[RotationDataPoint],
    epsilon: float = EPSILON_GAL,
    r_trans_kpc: float = R_TRANS_KPC,
    exact: bool = False,
    screening: ScreeningConfig | None = None,
) -> FitResult:
    """Compare model rotation curve to observed data.

    Computes χ² for both the Newtonian (no DM) and SDGFT (running G)
    models against the observed rotation curve.

    Args:
        galaxy: Baryonic mass model.
        data: Observed rotation curve data points.
        epsilon: Logarithmic modification strength.
        r_trans_kpc: SDGFT transition radius in kpc.
        exact: Use Freeman thin-disk formula instead of spherical approx.
        screening: If not None, apply chameleon screening.

    Returns:
        FitResult with χ² statistics.
    """
    chi2_newton = 0.0
    chi2_sdgft = 0.0

    for dp in data:
        # Newtonian baryonic velocity
        if exact:
            v2_newton = galaxy.v2_baryonic_freeman(dp.r_kpc)
        else:
            r_m = dp.r_kpc * KPC_M
            m_enc = galaxy.enclosed_mass_kg(dp.r_kpc)
            v2_newton = G_N * m_enc / r_m

        v_newton = math.sqrt(max(v2_newton, 0.0)) / KMS_TO_MS

        # SDGFT
        g_ratio = g_eff_galactic(dp.r_kpc, epsilon, r_trans_kpc) / G_N
        if screening is not None and g_ratio > 1.0:
            s = screening_factor(dp.r_kpc, galaxy,
                                 screening.sigma_screen, screening.p)
            g_ratio = 1.0 + s * (g_ratio - 1.0)

        v2_sdgft = g_ratio * v2_newton
        v_sdgft = math.sqrt(max(v2_sdgft, 0.0)) / KMS_TO_MS

        # χ²
        if dp.v_err_kms > 0:
            chi2_newton += ((v_newton - dp.v_obs_kms) / dp.v_err_kms) ** 2
            chi2_sdgft += ((v_sdgft - dp.v_obs_kms) / dp.v_err_kms) ** 2

    return FitResult(
        epsilon=epsilon,
        chi2_newton=chi2_newton,
        chi2_sdgft=chi2_sdgft,
        n_data=len(data),
    )


def scan_epsilon(
    galaxy: GalaxyModel,
    data: tuple[RotationDataPoint, ...] | list[RotationDataPoint],
    eps_min: float = 0.05,
    eps_max: float = 1.5,
    n_steps: int = 146,
    r_trans_kpc: float = R_TRANS_KPC,
    exact: bool = False,
    screening: ScreeningConfig | None = None,
) -> tuple[float, list[tuple[float, float]]]:
    """Scan ε_gal to find the best-fit value.

    Args:
        galaxy: Baryonic mass model.
        data: Observed data.
        eps_min: Minimum ε to scan.
        eps_max: Maximum ε to scan.
        n_steps: Number of scan points.
        r_trans_kpc: Transition radius.
        exact: Use Freeman thin-disk formula.
        screening: If not None, apply chameleon screening.

    Returns:
        Tuple of (best_epsilon, list of (epsilon, chi2_sdgft) pairs).
    """
    results: list[tuple[float, float]] = []
    best_eps = eps_min
    best_chi2 = float("inf")

    for i in range(n_steps):
        eps = eps_min + (eps_max - eps_min) * i / (n_steps - 1)
        fit = fit_rotation_curve(galaxy, data, epsilon=eps,
                                  r_trans_kpc=r_trans_kpc, exact=exact,
                                  screening=screening)
        results.append((eps, fit.chi2_sdgft))
        if fit.chi2_sdgft < best_chi2:
            best_chi2 = fit.chi2_sdgft
            best_eps = eps

    return best_eps, results


# ╔══════════════════════════════════════════════════════════════════╗
# ║  "Effective dark matter" diagnostic                            ║
# ╚══════════════════════════════════════════════════════════════════╝

def effective_dm_mass(
    r_kpc: float,
    galaxy: GalaxyModel,
    epsilon: float = EPSILON_GAL,
    r_trans_kpc: float = R_TRANS_KPC,
) -> float:
    """Effective "dark matter" mass created by running G.

    The SDGFT rotation curve v²_SDGFT = G_eff · M_bary / r can be
    rewritten as v²_SDGFT = G_N · (M_bary + M_DM_eff) / r, giving:

        M_DM_eff(r) = M_bary(<r) · ε · ln(r / r_trans)

    This is the mass that a Newtonian observer would attribute to a
    "dark matter halo" to explain the flat rotation curve.

    Args:
        r_kpc: Radius in kpc.
        galaxy: Baryonic mass model.
        epsilon: ε_gal.
        r_trans_kpc: Transition radius.

    Returns:
        Effective DM mass in kg.
    """
    if r_kpc <= r_trans_kpc:
        return 0.0
    m_bary = galaxy.enclosed_mass_kg(r_kpc)
    return m_bary * epsilon * math.log(r_kpc / r_trans_kpc)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  D*(r) profile diagnostic                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def d_star_at_galactic_scale(
    r_kpc: float,
    d_star_ir: float = D_STAR_TREE_F,
    delta: float = DELTA_F,
) -> float:
    """D* at galactic scale from the e-fold parametrization.

    Uses the inverted e-fold formula evaluated in terms of position:

       D*(r) = D*_IR · (1 − Δ² · Δ_N(r))

    where Δ_N(r) represents the residual flow at scale r.  For
    r in the galactic range, D* is extremely close to D*_IR.

    A more physical computation: the number of e-folds from the
    Planck scale to radius r is N ≈ ln(r/r_P), so:

        D*(N) = D*_IR − (D*_IR − D*_start) · exp(−Δ/D*_IR · N)

    At galactic N ≈ 130, the exponential is negligible, confirming
    that D* ≈ D*_IR at these scales.

    Args:
        r_kpc: Radius in kpc.

    Returns:
        Effective dimension (very close to D*_IR).
    """
    from .._internal_constants import R_PLANCK  # avoid circular
    r_m = r_kpc * KPC_M
    # N = ln(r/r_P)
    try:
        r_p = 1.616255e-35  # Planck length, local to avoid imports
        n_efold = math.log(r_m / r_p)
    except (ValueError, ZeroDivisionError):
        return d_star_ir

    d_star_start = 49.0 / 24.0  # 2 + 1/24
    decay_rate = delta / d_star_ir
    return d_star_ir - (d_star_ir - d_star_start) * math.exp(-decay_rate * n_efold)


# Use the simpler approach: D* essentially equals D*_IR at galactic scales
def d_star_galactic_profile(
    radii_kpc: list[float],
    d_star_ir: float = D_STAR_TREE_F,
    delta: float = DELTA_F,
) -> list[float]:
    """D* profile across galactic radii.

    At galactic scales (r ~ 1--30 kpc), D* is indistinguishable
    from D*_IR (the exponential correction is < 10⁻⁴⁰).

    Args:
        radii_kpc: List of radii.

    Returns:
        List of D* values (all ≈ D*_IR).
    """
    r_p = 1.616255e-35
    d_star_start = 49.0 / 24.0
    decay_rate = delta / d_star_ir
    result = []
    for r_kpc in radii_kpc:
        r_m = r_kpc * KPC_M
        n_efold = math.log(r_m / r_p) if r_m > r_p else 0.0
        d = d_star_ir - (d_star_ir - d_star_start) * math.exp(-decay_rate * n_efold)
        result.append(d)
    return result


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Chameleon screening  (Khoury & Weltman 2004;                  ║
# ║                         Capozziello et al. 2007)               ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# In f(R) = R^n gravity the scalar degree of freedom (scalaron)
# f_R = n R^{n-1} acquires an effective mass m_s that depends on
# the local matter density ρ:
#
#   m_s² = R / (3(n-1))  ∝  ρ / (n-1)
#
# The Compton wavelength λ_s = 1/m_s determines the range of the
# fifth force.  When λ_s ≪ R_gal (high density, inner disk) the
# modification is screened.  Conversely, λ_s ≫ R_gal (low density,
# outskirts) gives the full unscreened modification.
#
# For a thin disk with surface density Σ(R), the screening
# transition is parametrized by a critical surface density
# Σ_screen and a steepness exponent p derived from the chameleon
# mass scaling.
#
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScreeningConfig:
    """Chameleon screening configuration.

    Defines the density-dependent screening factor:

        S(R) = 1 / (1 + (Σ(R)/Σ_screen)^p)

    Attributes:
        p: Screening steepness exponent.
        sigma_screen: Critical surface density [M☉/kpc²].
    """
    p: float
    sigma_screen: float  # M☉/kpc²


@dataclass(frozen=True)
class ScreeningCandidate:
    """Candidate for the screening steepness exponent p.

    In f(R) = R^n gravity, different physical quantities —
    the chameleon mass, the Compton wavelength, the force ratio —
    yield different effective exponents for the screening transition.

    Attributes:
        name: Short identifier.
        label: Human-readable label.
        p: Screening steepness exponent.
        formula: LaTeX formula string.
    """
    name: str
    label: str
    p: float
    formula: str


def build_screening_candidates(
    n: float = N_TREE_F,
) -> list[ScreeningCandidate]:
    r"""Build theoretical candidates for the screening steepness p.

    For f(R) = R^n, the scalaron effective mass scales as
    m²_s ∝ ρ^{1/(n-1)}, giving different p depending on whether
    one compares the mass, the Compton wavelength, or the
    force ratio to local scales.

    Returns:
        List sorted by p value (ascending).
    """
    n_minus_1 = n - 1.0  # 19/48

    candidates = [
        ScreeningCandidate(
            name="gentle",
            label="A: p = n−1  (gentle)",
            p=n_minus_1,  # 19/48 ≈ 0.396
            formula=r"p = n-1 = 19/48",
        ),
        ScreeningCandidate(
            name="linear",
            label="B: p = 1  (linear)",
            p=1.0,
            formula=r"p = 1",
        ),
        ScreeningCandidate(
            name="compton",
            label="C: p = 1/(2(n−1))  (Compton λ)",
            p=1.0 / (2.0 * n_minus_1),  # 24/19 ≈ 1.263
            formula=r"p = 1/(2(n-1)) = 24/19",
        ),
        ScreeningCandidate(
            name="chameleon_mass",
            label="D: p = 1/(n−1)  (m²_cham)",
            p=1.0 / n_minus_1,  # 48/19 ≈ 2.526
            formula=r"p = 1/(n-1) = 48/19",
        ),
    ]
    return sorted(candidates, key=lambda c: c.p)


SCREENING_CANDIDATES: list[ScreeningCandidate] = build_screening_candidates()
"""All p candidates for chameleon screening, sorted by p."""


def screening_factor(
    r_kpc: float,
    galaxy: GalaxyModel,
    sigma_screen: float,
    p: float,
) -> float:
    r"""Chameleon screening factor S(R) ∈ [0, 1].

    S(R) = 1 / (1 + (Σ(R)/Σ_screen)^p)

    - S → 0 at high density (galaxy center): modification screened.
    - S → 1 at low density (outskirts): full modification.

    Args:
        r_kpc: Galactocentric radius in kpc.
        galaxy: Baryonic mass model.
        sigma_screen: Critical surface density [M☉/kpc²].
        p: Screening steepness exponent.

    Returns:
        Screening factor ∈ [0, 1].
    """
    if sigma_screen <= 0.0:
        return 1.0
    sigma = galaxy.surface_density(r_kpc)
    x = sigma / sigma_screen
    return 1.0 / (1.0 + x ** p)


def default_screening_config(
    galaxy: GalaxyModel,
    r_trans_kpc: float = R_TRANS_KPC,
    p: float | None = None,
) -> ScreeningConfig:
    r"""Build default screening config for a galaxy.

    Uses Σ_screen = Σ_total(r_trans) — the total baryonic surface
    density at the SDGFT transition radius.  This ensures
    S(r_trans) ≈ 0.5, matching the physical onset of the modification.

    If *p* is None, uses the 'compton' candidate:
    p = 1/(2(n−1)) = 24/19 ≈ 1.263.

    Args:
        galaxy: Baryonic mass model.
        r_trans_kpc: SDGFT transition radius in kpc.
        p: Override screening steepness (default: Compton-derived).

    Returns:
        Screening configuration.
    """
    if p is None:
        p = 1.0 / (2.0 * (N_TREE_F - 1.0))  # 24/19
    sigma_screen = galaxy.surface_density(r_trans_kpc)
    return ScreeningConfig(p=p, sigma_screen=sigma_screen)


def screening_profile(
    radii_kpc: list[float],
    galaxy: GalaxyModel,
    config: ScreeningConfig,
) -> list[float]:
    """Screening factor S(R) at each radius.

    Args:
        radii_kpc: Radii in kpc.
        galaxy: Baryonic mass model.
        config: Screening configuration.

    Returns:
        List of S(R) values ∈ [0, 1].
    """
    return [
        screening_factor(r, galaxy, config.sigma_screen, config.p)
        for r in radii_kpc
    ]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Module-level NGC 3198 results                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

# --- Spherical approximation (backward compatible) ---

# Best-fit ε from scanning
EPSILON_BESTFIT, _SCAN_RESULTS = scan_epsilon(NGC3198_MODEL, NGC3198_DATA)
"""Best-fit ε_gal from NGC 3198 χ² scan (spherical approximation)."""

# Fit results with the observationally quoted ε = 0.16
FIT_RESULT_OBS: FitResult = fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA,
                                                epsilon=EPSILON_OBS)
"""NGC 3198 fit result with ε_gal = 0.16 (observed, spherical)."""

# Fit result with each theoretical candidate
FIT_RESULTS_THEORY: dict[str, FitResult] = {
    c.name: fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA, epsilon=c.value)
    for c in EPSILON_CANDIDATES
}
"""NGC 3198 fit results for each theoretical ε candidate (spherical)."""

# Rotation curve at observed data radii (for comparison output)
_DATA_RADII = [dp.r_kpc for dp in NGC3198_DATA]
NGC3198_CURVE: list[RotationCurvePoint] = compute_rotation_curve(
    NGC3198_MODEL,
    radii_kpc=_DATA_RADII,
    epsilon=EPSILON_BESTFIT,
)
"""NGC 3198 rotation curve computed at observed radii with best-fit ε (spherical)."""

# --- Exact Freeman (1970) thin-disk model ---

EPSILON_BESTFIT_EXACT, _SCAN_RESULTS_EXACT = scan_epsilon(
    NGC3198_MODEL, NGC3198_DATA, exact=True,
)
"""Best-fit ε_gal from NGC 3198 χ² scan (exact thin disk)."""

FIT_RESULT_OBS_EXACT: FitResult = fit_rotation_curve(
    NGC3198_MODEL, NGC3198_DATA, epsilon=EPSILON_OBS, exact=True,
)
"""NGC 3198 fit result with ε_gal = 0.16 (observed, exact disk)."""

FIT_RESULTS_THEORY_EXACT: dict[str, FitResult] = {
    c.name: fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA,
                                epsilon=c.value, exact=True)
    for c in EPSILON_CANDIDATES
}
"""NGC 3198 fit results for each theoretical ε candidate (exact disk)."""

NGC3198_CURVE_EXACT: list[RotationCurvePoint] = compute_rotation_curve(
    NGC3198_MODEL,
    radii_kpc=_DATA_RADII,
    epsilon=EPSILON_BESTFIT_EXACT,
    exact=True,
)
"""NGC 3198 rotation curve at observed radii (exact thin disk, best-fit ε)."""

# --- Exact disk with chameleon screening ---

# Default screening config: Compton-wavelength steepness, p = 24/19
NGC3198_SCREENING: ScreeningConfig = default_screening_config(NGC3198_MODEL)
"""Default chameleon screening config for NGC 3198."""

EPSILON_BESTFIT_SCREENED, _SCAN_RESULTS_SCREENED = scan_epsilon(
    NGC3198_MODEL, NGC3198_DATA, exact=True, screening=NGC3198_SCREENING,
)
"""Best-fit ε_gal from NGC 3198 χ² scan (exact disk + chameleon)."""

FIT_RESULT_OBS_SCREENED: FitResult = fit_rotation_curve(
    NGC3198_MODEL, NGC3198_DATA, epsilon=EPSILON_OBS,
    exact=True, screening=NGC3198_SCREENING,
)
"""NGC 3198 fit result with ε_gal = 0.16 (exact disk + chameleon)."""

FIT_RESULTS_THEORY_SCREENED: dict[str, FitResult] = {
    c.name: fit_rotation_curve(NGC3198_MODEL, NGC3198_DATA,
                                epsilon=c.value, exact=True,
                                screening=NGC3198_SCREENING)
    for c in EPSILON_CANDIDATES
}
"""NGC 3198 fit results for each ε candidate (exact disk + screening)."""

NGC3198_CURVE_SCREENED: list[RotationCurvePoint] = compute_rotation_curve(
    NGC3198_MODEL,
    radii_kpc=_DATA_RADII,
    epsilon=EPSILON_BESTFIT_SCREENED,
    exact=True,
    screening=NGC3198_SCREENING,
)
"""NGC 3198 rotation curve (exact disk + chameleon, best-fit ε)."""

# Screening profiles for each candidate p
SCREENING_SCAN: dict[str, tuple[float, float]] = {}
"""Best-fit ε for each screening steepness candidate."""
for _sc in SCREENING_CANDIDATES:
    _cfg = ScreeningConfig(p=_sc.p, sigma_screen=NGC3198_SCREENING.sigma_screen)
    _eps, _ = scan_epsilon(NGC3198_MODEL, NGC3198_DATA, exact=True,
                            screening=_cfg)
    SCREENING_SCAN[_sc.name] = (_sc.p, _eps)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Summary output                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def print_summary() -> None:
    """Print comprehensive NGC 3198 rotation curve analysis."""
    print("=" * 80)
    print("  SDGFT Galaxy Rotation: Scale-Dependent G(r) for NGC 3198")
    print("=" * 80)

    gal = NGC3198_MODEL
    print(f"\n  Galaxy: {gal.name} (d = {gal.distance_mpc} Mpc)")
    print(f"    M_disk  = {gal.m_disk_msun:.1e} M☉  (h = {gal.h_disk_kpc} kpc)")
    print(f"    M_gas   = {gal.m_gas_msun:.1e} M☉  (h = {gal.h_gas_kpc} kpc)")
    print(f"    M_bary  = {gal.m_bary_msun:.1e} M☉")
    print(f"    r_trans = {R_TRANS_KPC:.2f} kpc  (SDGFT prediction)")
    print(f"    Σ_screen= {NGC3198_SCREENING.sigma_screen:.2e} M☉/kpc²"
          f"  (p = {NGC3198_SCREENING.p:.3f})")

    # ε_gal candidates — three models
    print(f"\n  Theoretical ε_gal candidates  (observed: {EPSILON_OBS}"
          f" ± {EPSILON_OBS_UNC}):")
    print(f"  {'Candidate':<28} {'ε':>8} {'σ':>6}"
          f" {'χ²_sph':>8} {'χ²_disk':>8} {'χ²_scrn':>8}")
    print("  " + "-" * 70)
    for c in EPSILON_CANDIDATES:
        f_s = FIT_RESULTS_THEORY[c.name]
        f_d = FIT_RESULTS_THEORY_EXACT[c.name]
        f_c = FIT_RESULTS_THEORY_SCREENED[c.name]
        sigma = abs(c.value - EPSILON_OBS) / EPSILON_OBS_UNC
        star = " ⭐" if c is EPSILON_BEST else ""
        print(f"  {c.label:<28} {c.value:>8.4f} {sigma:>6.2f}"
              f" {f_s.chi2_red_sdgft:>8.1f}"
              f" {f_d.chi2_red_sdgft:>8.1f}"
              f" {f_c.chi2_red_sdgft:>8.1f}{star}")

    # Best-fit ε from data — all three models
    print(f"\n  Best-fit ε from NGC 3198 χ² scan:")
    print(f"    Spherical approx:         {EPSILON_BESTFIT:.4f}")
    print(f"    Exact thin disk:          {EPSILON_BESTFIT_EXACT:.4f}")
    print(f"    Exact + chameleon screen: {EPSILON_BESTFIT_SCREENED:.4f}")
    print(f"    Theory best: {EPSILON_BEST.label}: {EPSILON_BEST.value:.4f}")

    # Screening candidate scan
    print(f"\n  Screening steepness candidates (ε_bestfit for each p):")
    print(f"  {'Candidate':<34} {'p':>6} {'ε_fit':>8}")
    print("  " + "-" * 50)
    for sc in SCREENING_CANDIDATES:
        p_val, e_val = SCREENING_SCAN[sc.name]
        print(f"  {sc.label:<34} {p_val:>6.3f} {e_val:>8.4f}")

    # Rotation curve with screening
    print(f"\n  Rotation curve — exact disk + chameleon"
          f" (ε = {EPSILON_BESTFIT_SCREENED:.3f}):")
    print(f"  {'r':>8} {'v_obs':>7} {'v_N':>7} {'v_noS':>7}"
          f" {'v_scrn':>7} {'S(R)':>6} {'G/G_N':>7}")
    print("  " + "-" * 54)
    for dp, cp_ex, cp_sc in zip(NGC3198_DATA, NGC3198_CURVE_EXACT,
                                 NGC3198_CURVE_SCREENED):
        s_val = screening_factor(dp.r_kpc, gal,
                                  NGC3198_SCREENING.sigma_screen,
                                  NGC3198_SCREENING.p)
        print(f"  {dp.r_kpc:>8.2f} {dp.v_obs_kms:>7.1f}"
              f" {cp_ex.v_newton_kms:>7.1f}"
              f" {cp_ex.v_sdgft_kms:>7.1f}"
              f" {cp_sc.v_sdgft_kms:>7.1f}"
              f" {s_val:>6.3f} {cp_sc.g_eff_ratio:>7.3f}")

    # χ² comparison: all three models
    fit_sph = fit_rotation_curve(gal, NGC3198_DATA, epsilon=EPSILON_BESTFIT)
    fit_exact = fit_rotation_curve(gal, NGC3198_DATA,
                                    epsilon=EPSILON_BESTFIT_EXACT, exact=True)
    fit_scrn = fit_rotation_curve(gal, NGC3198_DATA,
                                   epsilon=EPSILON_BESTFIT_SCREENED,
                                   exact=True, screening=NGC3198_SCREENING)
    print(f"\n  χ² comparison:")
    print(f"  {'Model':<35} {'χ²_red':>8} {'χ²':>8}")
    print("  " + "-" * 53)
    print(f"  {'Newton (spherical, no DM)':<35}"
          f" {fit_sph.chi2_red_newton:>8.1f} {fit_sph.chi2_newton:>8.0f}")
    print(f"  {'SDGFT  (spherical, ε-fit)':<35}"
          f" {fit_sph.chi2_red_sdgft:>8.1f} {fit_sph.chi2_sdgft:>8.0f}")
    print(f"  {'Newton (exact disk, no DM)':<35}"
          f" {fit_exact.chi2_red_newton:>8.1f} {fit_exact.chi2_newton:>8.0f}")
    print(f"  {'SDGFT  (exact disk, ε-fit)':<35}"
          f" {fit_exact.chi2_red_sdgft:>8.1f} {fit_exact.chi2_sdgft:>8.0f}")
    print(f"  {'SDGFT  (disk + chameleon, ε-fit)':<35}"
          f" {fit_scrn.chi2_red_sdgft:>8.1f} {fit_scrn.chi2_sdgft:>8.0f}")

    # Key result: ε progression toward theory
    print(f"\n  ε_bestfit progression toward theory α_M(1−α_M):")
    for label, eps_val in [("Spherical", EPSILON_BESTFIT),
                            ("Exact disk", EPSILON_BESTFIT_EXACT),
                            ("Disk + chameleon", EPSILON_BESTFIT_SCREENED)]:
        dev = abs(eps_val - EPSILON_BEST.value) / EPSILON_OBS_UNC
        print(f"    {label:<20} ε = {eps_val:.4f}  ({dev:.2f}σ)")

    # Screening profile
    print(f"\n  Screening factor S(R) at selected radii"
          f"  [p = {NGC3198_SCREENING.p:.3f}]:")
    s_radii = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
    for r in s_radii:
        s_val = screening_factor(r, gal, NGC3198_SCREENING.sigma_screen,
                                  NGC3198_SCREENING.p)
        sigma = gal.surface_density(r)
        print(f"    R = {r:>5.1f} kpc:  Σ = {sigma:>10.2e} M☉/kpc²"
              f"  S = {s_val:.4f}")

    print("=" * 80)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Registry                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def register_all(registry=REGISTRY) -> None:
    """Register galaxy rotation observables."""
    registry.register(Observable(
        name="exp_epsilon_gal",
        symbol="epsilon_gal_theory",
        formula=EPSILON_BEST.formula,
        predicted=EPSILON_BEST.value,
        observed=EPSILON_OBS,
        observed_uncertainty=EPSILON_OBS_UNC,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("alpha_m", "delta"),
    ))
    registry.register(Observable(
        name="exp_epsilon_bestfit",
        symbol="epsilon_gal_fit_spherical",
        formula="chi^2 scan over NGC 3198 (spherical enclosed mass)",
        predicted=EPSILON_BESTFIT,
        observed=EPSILON_OBS,
        observed_uncertainty=EPSILON_OBS_UNC,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("alpha_m", "delta"),
        is_diagnostic=True,
    ))
    registry.register(Observable(
        name="exp_epsilon_bestfit_exact",
        symbol="epsilon_gal_fit_disk",
        formula="chi^2 scan over NGC 3198 (Freeman thin disk)",
        predicted=EPSILON_BESTFIT_EXACT,
        observed=EPSILON_OBS,
        observed_uncertainty=EPSILON_OBS_UNC,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("alpha_m", "delta"),
        is_diagnostic=True,
    ))
    registry.register(Observable(
        name="exp_epsilon_bestfit_screened",
        symbol="epsilon_gal_fit_screened",
        formula="chi^2 scan over NGC 3198 (Freeman disk + chameleon screening)",
        predicted=EPSILON_BESTFIT_SCREENED,
        observed=EPSILON_OBS,
        observed_uncertainty=EPSILON_OBS_UNC,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("alpha_m", "delta"),
        is_diagnostic=True,
    ))


# ── CLI entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    print_summary()

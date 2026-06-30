"""SPARC Batch Processing — Global test of SDGFT galaxy rotation.

Processes all 175 galaxies from the SPARC database (Lelli, McGaugh &
Schombert 2016) to test whether the SDGFT scale-dependent G(r)
reproduces observed rotation curves with a *universal* coupling
constant ε.

Physics
-------
For each galaxy at radius R > r_trans the effective gravitational
constant is

    G_eff(R) = G_N · [1 + ε · ln(R / r_trans)]

The baryonic velocity is computed from SPARC decomposed components
(at stellar M/L = Υ_disk, Υ_bul):

    v²_bar(R) = Υ_disk · V²_disk + Υ_bul · V²_bul + V²_gas

The SDGFT-modified velocity is

    v²_SDGFT(R) = (G_eff / G_N) · S(R) · v²_bar + [1 − S(R)] · v²_bar

where S(R) is the chameleon screening factor for each galaxy.

The fit determines the optimal ε for each galaxy via χ² minimisation.
The key prediction:  ⟨ε⟩ over all 175 galaxies ≈ α_M(1−α_M) ≈ 0.172.

Data source
-----------
SPARC: Spitzer Photometry and Accurate Rotation Curves
    Lelli F., McGaugh S.S., Schombert J.M.  (2016, AJ, 152, 157)
    http://astroweb.cwru.edu/SPARC/1

Default path: ``/home/david/Coding/data/sparc/``

Exports
-------
- SPARCDataPoint       — per-radius data from rotation models
- SPARCGalaxy          — full galaxy with catalog + rotation data
- GalaxyFitResult      — per-galaxy ε fit result
- BatchResult          — full 175-galaxy batch analysis
- load_sparc_catalog   — parse SPARC_Lelli2016c.mrt
- load_rotation_curve  — parse one *_rotmod.dat file
- load_sparc_database  — load all 175 galaxies
- fit_galaxy           — fit ε for one galaxy (with screening)
- run_batch            — fit all galaxies, compute statistics
- SPARC_BATCH          — module-level pre-computed BatchResult
- print_summary        — formatted output
- register_all         — register ⟨ε⟩ observable
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from sdgft.gravity import ALPHA_M_TREE_F
from sdgft.physical_constants import G_N
from sdgft.registry import REGISTRY, Observable
from sdgft.tully_fisher import R_TRANS_KPC

# ── Constants ─────────────────────────────────────────────────────

#: Default SPARC data directory.
SPARC_DATA_DIR: str = "/home/david/Coding/data/sparc"

#: Default stellar M/L at 3.6μm (Schombert & McGaugh 2014).
ML_DISK: float = 0.5
ML_BULGE: float = 0.7

#: Theoretical prediction for ε from SDGFT.
EPSILON_THEORY: float = ALPHA_M_TREE_F * (1.0 - ALPHA_M_TREE_F)

#: ε scan parameters.
EPSILON_MIN: float = 0.01
EPSILON_MAX: float = 5.0
N_EPSILON_STEPS: int = 500

#: Minimum data points required for a reliable fit.
MIN_DATA_POINTS: int = 5

#: Quality flag filter: 1=high, 2=medium, 3=low.
#  We accept Q ≤ 2 by default (high + medium quality).
MAX_QUALITY: int = 3

# ── Resummed power-law running ────────────────────────────────────
#
# The first-order formula  G_eff = G_N [1 + ε·ln(R/r_t)]  is the
# leading Taylor term of the exact RG resummation:
#
#     G_eff(R) = G_N · (R / r_trans)^ν      (R > r_trans)
#
# where ν = 2α_M = 2·(19/86) = 19/43 ≈ 0.4419.
#
# Physics:  G_N ∝ M_P^{−2}.  The Planck mass M_P(r) runs with
# anomalous dimension α_M in the IR:  M_P(r) ∝ r^{−α_M}.
# Therefore  G(r) ∝ r^{2α_M}.
#
# First-order expansion:
#   (R/r_t)^ν ≈ 1 + ν·ln(R/r_t) + ½ν²·ln²(R/r_t) + ···
#
# At R = 44 kpc the expansion parameter ν·ln(R/r_t) ≈ 1.7 > 1,
# making the first-order truncation invalid.
# The power-law correctly sums all orders.
#
# Connection to Tully-Fisher:
#   b_TF = D* + 1 = 91/24  (mass–velocity exponent)
#   ν = 2α_M = 19/43       (radial G(r) exponent)
#   Both derive from D* = 67/24.

#: Resummed power-law exponent: ν = 2α_M = 19/43 ≈ 0.4419.
NU_THEORY: float = 2.0 * ALPHA_M_TREE_F

#: ν scan parameters for fitted power-law mode.
NU_MIN: float = 0.01
NU_MAX: float = 2.0
N_NU_STEPS: int = 200


# ── Data structures ───────────────────────────────────────────────

@dataclass(frozen=True)
class SPARCCatalogEntry:
    """One row from SPARC_Lelli2016c.mrt (galaxy properties)."""

    name: str                # Galaxy identifier
    hubble_type: int         # Hubble type code (0–11)
    distance_mpc: float      # Distance [Mpc]
    e_distance_mpc: float    # Error on distance [Mpc]
    distance_method: int     # Distance method flag
    inclination_deg: float   # Inclination [deg]
    e_inclination_deg: float # Error on inclination [deg]
    luminosity_1e9Lsun: float  # Total 3.6μm luminosity [10⁹ L☉]
    e_luminosity_1e9Lsun: float
    r_eff_kpc: float         # Effective radius [kpc]
    sb_eff: float            # Effective surface brightness [L☉/pc²]
    r_disk_kpc: float        # Disk scale length [kpc]
    sb_disk: float           # Disk central surface brightness [L☉/pc²]
    m_hi_1e9Msun: float      # Total HI mass [10⁹ M☉]
    r_hi_kpc: float          # HI radius [kpc]
    v_flat_kms: float        # Flat rotation velocity [km/s]
    e_v_flat_kms: float      # Error on Vflat [km/s]
    quality: int             # Quality flag (1=high, 2=medium, 3=low)


@dataclass(frozen=True)
class SPARCDataPoint:
    """One radial point from a SPARC rotation model file."""

    r_kpc: float         # Galactocentric radius [kpc]
    v_obs_kms: float     # Observed circular velocity [km/s]
    v_err_kms: float     # Uncertainty in Vobs [km/s]
    v_gas_kms: float     # Gas velocity contribution [km/s]
    v_disk_kms: float    # Disk velocity contribution (M/L=1) [km/s]
    v_bul_kms: float     # Bulge velocity contribution (M/L=1) [km/s]
    sb_disk: float       # Disk surface brightness [L☉/pc²]
    sb_bul: float        # Bulge surface brightness [L☉/pc²]


@dataclass(frozen=True)
class SPARCGalaxy:
    """A fully loaded SPARC galaxy with catalog info + rotation data."""

    catalog: SPARCCatalogEntry
    data: tuple[SPARCDataPoint, ...]

    @property
    def name(self) -> str:
        return self.catalog.name

    @property
    def n_points(self) -> int:
        return len(self.data)

    @property
    def r_max_kpc(self) -> float:
        if not self.data:
            return 0.0
        return self.data[-1].r_kpc


@dataclass(frozen=True)
class GalaxyFitResult:
    """Fit result for a single SPARC galaxy."""

    name: str              # Galaxy identifier
    epsilon_best: float    # Best-fit ε
    chi2_newton: float     # χ² for pure Newtonian (no DM)
    chi2_sdgft: float      # χ² for SDGFT at ε_best
    n_data: int            # Number of data points used
    quality: int           # SPARC quality flag
    distance_mpc: float    # Distance
    v_flat_kms: float      # Flat velocity
    r_max_kpc: float       # Maximum radius in data
    converged: bool        # Did scan find a minimum (not at boundary)?

    @property
    def chi2_red_newton(self) -> float:
        dof = max(self.n_data - 1, 1)
        return self.chi2_newton / dof

    @property
    def chi2_red_sdgft(self) -> float:
        dof = max(self.n_data - 2, 1)
        return self.chi2_sdgft / dof

    @property
    def improvement(self) -> float:
        """χ²_Newton / χ²_SDGFT."""
        if self.chi2_sdgft <= 0:
            return float("inf")
        return self.chi2_newton / self.chi2_sdgft


@dataclass(frozen=True)
class BatchResult:
    """Aggregate results from processing all SPARC galaxies."""

    fits: tuple[GalaxyFitResult, ...]

    # ── Sample statistics ──

    n_total: int           # Total galaxies loaded
    n_fitted: int          # Galaxies with valid fits
    n_converged: int       # Fits that converged (not at boundary)
    n_improved: int        # Galaxies where SDGFT beats Newton

    # ── ε statistics (converged galaxies only) ──

    epsilon_mean: float            # Unweighted mean of ε_best
    epsilon_median: float          # Median ε_best
    epsilon_std: float             # Standard deviation
    epsilon_sem: float             # Standard error of the mean
    epsilon_weighted_mean: float   # Weighted by 1/χ²_red
    epsilon_weighted_sem: float    # Weighted SEM

    # ── Comparison to theory ──

    epsilon_theory: float          # α_M(1−α_M)
    sigma_from_theory: float       # |⟨ε⟩ − ε_theory| / SEM
    sigma_weighted: float          # Same for weighted mean

    # ── χ² summary ──

    median_chi2_red_newton: float
    median_chi2_red_sdgft: float

    # ── Quality breakdown ──

    n_quality_1: int
    n_quality_2: int
    n_quality_3: int

    # ── Optional: per-quality ε ──

    epsilon_mean_q1: float         # Mean ε for Q=1 galaxies only
    epsilon_mean_q2: float         # Mean ε for Q=2 galaxies only
    n_q1_converged: int
    n_q2_converged: int

    # ── M/L sensitivity ──

    ml_scan: tuple[tuple[float, float, float], ...]  # (M/L, ⟨ε⟩, SEM)


@dataclass(frozen=True)
class PowerlawFitResult:
    """Fit result for one galaxy with power-law G_eff(R) = G_N·(R/r_t)^ν."""

    name: str
    nu_best: float           # Best-fit ν (fitted)
    chi2_newton: float       # χ² for pure baryonic Newton
    chi2_powerlaw_fit: float # χ² at best-fit ν
    chi2_zero_param: float   # χ² at ν = 2α_M (zero free parameters)
    n_data: int
    quality: int
    distance_mpc: float
    v_flat_kms: float
    r_max_kpc: float
    converged: bool          # Fitted ν not at scan boundary

    @property
    def chi2_red_newton(self) -> float:
        return self.chi2_newton / max(self.n_data - 1, 1)

    @property
    def chi2_red_powerlaw_fit(self) -> float:
        return self.chi2_powerlaw_fit / max(self.n_data - 2, 1)

    @property
    def chi2_red_zero_param(self) -> float:
        # Zero free parameters from SDGFT → same DOF as Newton.
        return self.chi2_zero_param / max(self.n_data - 1, 1)

    @property
    def improvement_fit(self) -> float:
        """χ²_Newton / χ²_powerlaw(fitted ν)."""
        if self.chi2_powerlaw_fit <= 0:
            return float("inf")
        return self.chi2_newton / self.chi2_powerlaw_fit

    @property
    def improvement_zero(self) -> float:
        """χ²_Newton / χ²_powerlaw(ν = 2α_M)."""
        if self.chi2_zero_param <= 0:
            return float("inf")
        return self.chi2_newton / self.chi2_zero_param


@dataclass(frozen=True)
class PowerlawBatchResult:
    """Aggregate results from power-law SPARC analysis."""

    fits: tuple[PowerlawFitResult, ...]

    n_total: int
    n_converged: int
    n_improved_fit: int    # SDGFT(fitted ν) beats Newton
    n_improved_zero: int   # SDGFT(ν = 2α_M) beats Newton

    # ── ν statistics (converged galaxies only) ──

    nu_mean: float
    nu_median: float
    nu_std: float
    nu_sem: float
    nu_weighted_mean: float
    nu_weighted_sem: float

    # ── Theory comparison ──

    nu_theory: float
    sigma_from_theory: float
    sigma_weighted: float

    # ── χ² medians (full sample) ──

    median_chi2_red_newton: float
    median_chi2_red_powerlaw_fit: float
    median_chi2_red_powerlaw_zero: float


# ── Data loading ──────────────────────────────────────────────────

def load_sparc_catalog(
    data_dir: str = SPARC_DATA_DIR,
) -> list[SPARCCatalogEntry]:
    """Parse SPARC_Lelli2016c.mrt → list of catalog entries.

    The .mrt file has fixed-width columns as described in the header.
    Data rows start after the last ``---`` separator line.
    """
    filepath = os.path.join(data_dir, "SPARC_Lelli2016c.mrt")
    entries: list[SPARCCatalogEntry] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        all_lines = fh.readlines()

    # Find last separator line (starts with "---")
    last_sep = 0
    for i, line in enumerate(all_lines):
        if line.startswith("---"):
            last_sep = i

    # Parse data rows after the last separator
    for line in all_lines[last_sep + 1:]:
        stripped = line.rstrip("\n")
        if len(stripped) < 90:
            continue

        # Whitespace-split approach: each data row has consistent columns
        parts = stripped.split()
        if len(parts) < 18:
            continue

        # The galaxy name may contain hyphens but no spaces in the MRT
        # Columns: name T D e_D f_D Inc e_Inc L e_L Reff SBeff Rdisk ...
        try:
            name = parts[0]
            hubble_type = int(parts[1])
            distance = float(parts[2])
            e_distance = float(parts[3])
            dist_method = int(parts[4])
            inc = float(parts[5])
            e_inc = float(parts[6])
            lum = float(parts[7])
            e_lum = float(parts[8])
            r_eff = float(parts[9])
            sb_eff = float(parts[10])
            r_disk = float(parts[11])
            sb_disk = float(parts[12])
            m_hi = float(parts[13])
            r_hi = float(parts[14])
            v_flat = float(parts[15])
            e_v_flat = float(parts[16])
            quality = int(parts[17])
        except (ValueError, IndexError):
            continue

        entries.append(SPARCCatalogEntry(
            name=name,
            hubble_type=hubble_type,
            distance_mpc=distance,
            e_distance_mpc=e_distance,
            distance_method=dist_method,
            inclination_deg=inc,
            e_inclination_deg=e_inc,
            luminosity_1e9Lsun=lum,
            e_luminosity_1e9Lsun=e_lum,
            r_eff_kpc=r_eff,
            sb_eff=sb_eff,
            r_disk_kpc=r_disk,
            sb_disk=sb_disk,
            m_hi_1e9Msun=m_hi,
            r_hi_kpc=r_hi,
            v_flat_kms=v_flat,
            e_v_flat_kms=e_v_flat,
            quality=quality,
        ))

    return entries


def load_rotation_curve(
    galaxy_name: str,
    data_dir: str = SPARC_DATA_DIR,
) -> tuple[SPARCDataPoint, ...]:
    """Load one *_rotmod.dat file → tuple of data points.

    Parameters
    ----------
    galaxy_name : str
        Galaxy identifier matching the filename (e.g. "NGC3198").
    data_dir : str
        Path to the SPARC data directory containing Rotmod_LTG/.

    Returns
    -------
    tuple[SPARCDataPoint, ...]
        Parsed data points sorted by radius.
    """
    filepath = os.path.join(data_dir, "Rotmod_LTG",
                             f"{galaxy_name}_rotmod.dat")
    points: list[SPARCDataPoint] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue

            points.append(SPARCDataPoint(
                r_kpc=float(parts[0]),
                v_obs_kms=float(parts[1]),
                v_err_kms=float(parts[2]),
                v_gas_kms=float(parts[3]),
                v_disk_kms=float(parts[4]),
                v_bul_kms=float(parts[5]),
                sb_disk=float(parts[6]),
                sb_bul=float(parts[7]),
            ))

    return tuple(sorted(points, key=lambda p: p.r_kpc))


def load_sparc_database(
    data_dir: str = SPARC_DATA_DIR,
    max_quality: int = MAX_QUALITY,
) -> list[SPARCGalaxy]:
    """Load all SPARC galaxies from catalog + rotation model files.

    Parameters
    ----------
    data_dir : str
        Path to SPARC data directory.
    max_quality : int
        Maximum quality flag to include (1=high only, 2=+medium, 3=all).

    Returns
    -------
    list[SPARCGalaxy]
        All loaded galaxies, sorted by name.
    """
    catalog = load_sparc_catalog(data_dir)
    galaxies: list[SPARCGalaxy] = []

    for entry in catalog:
        if entry.quality > max_quality:
            continue

        try:
            data = load_rotation_curve(entry.name, data_dir)
        except FileNotFoundError:
            continue

        if len(data) < MIN_DATA_POINTS:
            continue

        galaxies.append(SPARCGalaxy(catalog=entry, data=data))

    return sorted(galaxies, key=lambda g: g.name)


# ── Physics: SDGFT velocity from SPARC decomposition ─────────────

def v2_baryonic_sparc(
    dp: SPARCDataPoint,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """Baryonic v² from SPARC decomposed components.

    v²_bar = Υ_disk · V²_disk + Υ_bul · V²_bul + |V_gas| · V_gas

    The gas term preserves sign (negative V_gas = inflow, rare).
    """
    v2_gas = abs(dp.v_gas_kms) * dp.v_gas_kms  # sign-preserved
    v2_disk = ml_disk * dp.v_disk_kms ** 2
    v2_bul = ml_bulge * dp.v_bul_kms ** 2
    return v2_disk + v2_bul + v2_gas


def v_sdgft_sparc(
    dp: SPARCDataPoint,
    epsilon: float,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """SDGFT-predicted velocity at one SPARC data point.

    Returns v in km/s.  G_eff/G_N factor applied to v²_bar.
    """
    v2_bar = v2_baryonic_sparc(dp, ml_disk, ml_bulge)

    if dp.r_kpc <= r_trans_kpc or v2_bar <= 0:
        return math.sqrt(max(v2_bar, 0.0))

    g_ratio = 1.0 + epsilon * math.log(dp.r_kpc / r_trans_kpc)
    v2_sdgft = g_ratio * v2_bar
    return math.sqrt(max(v2_sdgft, 0.0))


def v_newton_sparc(
    dp: SPARCDataPoint,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """Newtonian velocity from SPARC baryons (no DM, no SDGFT)."""
    v2 = v2_baryonic_sparc(dp, ml_disk, ml_bulge)
    return math.sqrt(max(v2, 0.0))


# ── Surface density estimate for screening ────────────────────────

def _estimate_surface_density(
    dp: SPARCDataPoint,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """Estimate Σ [M☉/kpc²] from SPARC surface brightness.

    Σ = Υ_disk · SB_disk + Υ_bul · SB_bul

    SB in L☉/pc² → multiply by 10⁶ to get L☉/kpc², then by M/L.
    """
    sigma = (ml_disk * dp.sb_disk + ml_bulge * dp.sb_bul) * 1e6
    return sigma


def _screening_factor_sparc(
    dp: SPARCDataPoint,
    sigma_screen: float,
    p: float,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """Chameleon screening factor S ∈ [0, 1] from SPARC surface brightness.

    S(R) = 1 / (1 + (Σ(R) / Σ_screen)^p)
    """
    if sigma_screen <= 0:
        return 1.0
    sigma = _estimate_surface_density(dp, ml_disk, ml_bulge)
    ratio = sigma / sigma_screen
    return 1.0 / (1.0 + ratio ** p)


def _estimate_sigma_screen(
    data: tuple[SPARCDataPoint, ...],
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """Estimate Σ_screen = Σ(r_trans) by interpolating SPARC SB data."""
    if not data:
        return 1.0

    # Find the two data points bracketing r_trans
    below = None
    above = None
    for dp in data:
        if dp.r_kpc <= r_trans_kpc:
            below = dp
        elif above is None:
            above = dp

    if below is not None and above is not None:
        # Linear interpolation of SB
        frac = ((r_trans_kpc - below.r_kpc)
                / max(above.r_kpc - below.r_kpc, 1e-10))
        sb_disk = below.sb_disk + frac * (above.sb_disk - below.sb_disk)
        sb_bul = below.sb_bul + frac * (above.sb_bul - below.sb_bul)
        sigma = (ml_disk * sb_disk + ml_bulge * sb_bul) * 1e6
        return max(sigma, 1.0)

    # Fallback: use nearest point
    nearest = min(data, key=lambda p: abs(p.r_kpc - r_trans_kpc))
    return max(_estimate_surface_density(nearest, ml_disk, ml_bulge), 1.0)


# ── Power-law (resummed) velocity ─────────────────────────────────

def v_sdgft_powerlaw(
    dp: SPARCDataPoint,
    nu: float,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """SDGFT velocity with resummed power-law G_eff.

    G_eff(R) = G_N · (R / r_trans)^ν   for R > r_trans.

    Returns v in km/s.
    """
    v2_bar = v2_baryonic_sparc(dp, ml_disk, ml_bulge)

    if dp.r_kpc <= r_trans_kpc or v2_bar <= 0:
        return math.sqrt(max(v2_bar, 0.0))

    g_ratio = (dp.r_kpc / r_trans_kpc) ** nu
    return math.sqrt(max(g_ratio * v2_bar, 0.0))


def _chi2_galaxy_powerlaw(
    data: tuple[SPARCDataPoint, ...],
    nu: float,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
    screening: bool = True,
    sigma_screen: float = 0.0,
    p_screen: float = 24.0 / 19.0,
) -> tuple[float, float]:
    """Compute χ²_Newton and χ²_powerlaw for one galaxy.

    Returns (chi2_newton, chi2_powerlaw).
    """
    chi2_n = 0.0
    chi2_s = 0.0

    for dp in data:
        if dp.v_err_kms <= 0:
            continue

        w = 1.0 / dp.v_err_kms ** 2

        # Newtonian
        v_n = v_newton_sparc(dp, ml_disk, ml_bulge)
        chi2_n += w * (dp.v_obs_kms - v_n) ** 2

        # Power-law SDGFT
        v2_bar = v2_baryonic_sparc(dp, ml_disk, ml_bulge)
        if dp.r_kpc <= r_trans_kpc or v2_bar <= 0:
            v_s = math.sqrt(max(v2_bar, 0.0))
        else:
            g_ratio = (dp.r_kpc / r_trans_kpc) ** nu

            # Apply chameleon screening
            if screening and g_ratio > 1.0 and sigma_screen > 0:
                s = _screening_factor_sparc(dp, sigma_screen, p_screen,
                                             ml_disk, ml_bulge)
                g_ratio = 1.0 + s * (g_ratio - 1.0)

            v_s = math.sqrt(max(g_ratio * v2_bar, 0.0))

        chi2_s += w * (dp.v_obs_kms - v_s) ** 2

    return chi2_n, chi2_s


# ── Fitting ───────────────────────────────────────────────────────

#: Default screening steepness (Compton candidate: p = 24/19).
P_SCREEN_DEFAULT: float = 24.0 / 19.0


def _chi2_galaxy(
    data: tuple[SPARCDataPoint, ...],
    epsilon: float,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
    screening: bool = True,
    sigma_screen: float = 0.0,
    p_screen: float = P_SCREEN_DEFAULT,
) -> tuple[float, float]:
    """Compute χ²_Newton and χ²_SDGFT for one galaxy.

    Returns (chi2_newton, chi2_sdgft).
    """
    chi2_n = 0.0
    chi2_s = 0.0

    for dp in data:
        if dp.v_err_kms <= 0:
            continue

        w = 1.0 / dp.v_err_kms ** 2

        # Newtonian
        v_n = v_newton_sparc(dp, ml_disk, ml_bulge)
        chi2_n += w * (dp.v_obs_kms - v_n) ** 2

        # SDGFT
        v2_bar = v2_baryonic_sparc(dp, ml_disk, ml_bulge)
        if dp.r_kpc <= r_trans_kpc or v2_bar <= 0:
            v_s = math.sqrt(max(v2_bar, 0.0))
        else:
            g_ratio = 1.0 + epsilon * math.log(dp.r_kpc / r_trans_kpc)

            # Apply chameleon screening
            if screening and g_ratio > 1.0 and sigma_screen > 0:
                s = _screening_factor_sparc(dp, sigma_screen, p_screen,
                                             ml_disk, ml_bulge)
                g_ratio = 1.0 + s * (g_ratio - 1.0)

            v_s = math.sqrt(max(g_ratio * v2_bar, 0.0))

        chi2_s += w * (dp.v_obs_kms - v_s) ** 2

    return chi2_n, chi2_s


def fit_galaxy(
    galaxy: SPARCGalaxy,
    eps_min: float = EPSILON_MIN,
    eps_max: float = EPSILON_MAX,
    n_steps: int = N_EPSILON_STEPS,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
    screening: bool = True,
    p_screen: float = P_SCREEN_DEFAULT,
) -> GalaxyFitResult:
    """Fit optimal ε for one SPARC galaxy via χ² scan.

    Parameters
    ----------
    galaxy : SPARCGalaxy
        Galaxy to fit.
    eps_min, eps_max : float
        ε scan range.
    n_steps : int
        Number of ε values to scan.
    screening : bool
        Whether to apply chameleon screening.
    p_screen : float
        Screening steepness exponent.

    Returns
    -------
    GalaxyFitResult
    """
    data = galaxy.data

    # Estimate Σ_screen for this galaxy
    sigma_screen = 0.0
    if screening:
        sigma_screen = _estimate_sigma_screen(
            data, r_trans_kpc, ml_disk, ml_bulge)

    # Scan ε
    best_eps = eps_min
    best_chi2_s = float("inf")
    chi2_n_stored = 0.0

    step = (eps_max - eps_min) / max(n_steps - 1, 1)
    for i in range(n_steps):
        eps = eps_min + i * step
        chi2_n, chi2_s = _chi2_galaxy(
            data, eps, r_trans_kpc, ml_disk, ml_bulge,
            screening, sigma_screen, p_screen,
        )
        if chi2_s < best_chi2_s:
            best_chi2_s = chi2_s
            best_eps = eps
            chi2_n_stored = chi2_n

    # Check convergence: not at boundary
    converged = (best_eps > eps_min + step and
                 best_eps < eps_max - step)

    return GalaxyFitResult(
        name=galaxy.name,
        epsilon_best=best_eps,
        chi2_newton=chi2_n_stored,
        chi2_sdgft=best_chi2_s,
        n_data=len(data),
        quality=galaxy.catalog.quality,
        distance_mpc=galaxy.catalog.distance_mpc,
        v_flat_kms=galaxy.catalog.v_flat_kms,
        r_max_kpc=galaxy.r_max_kpc,
        converged=converged,
    )


# ── Batch processing ─────────────────────────────────────────────

def _median(values: list[float]) -> float:
    """Compute median of a sorted list."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _weighted_mean_sem(
    values: list[float],
    weights: list[float],
) -> tuple[float, float]:
    """Weighted mean and standard error of the weighted mean."""
    if not values or not weights:
        return 0.0, 0.0
    w_sum = sum(weights)
    if w_sum <= 0:
        return 0.0, 0.0
    w_mean = sum(v * w for v, w in zip(values, weights)) / w_sum

    # Weighted SEM: 1/sqrt(sum_of_weights)
    w_sem = 1.0 / math.sqrt(w_sum) if w_sum > 0 else 0.0
    return w_mean, w_sem


def run_batch(
    data_dir: str = SPARC_DATA_DIR,
    max_quality: int = MAX_QUALITY,
    eps_min: float = EPSILON_MIN,
    eps_max: float = EPSILON_MAX,
    n_steps: int = N_EPSILON_STEPS,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
    screening: bool = True,
    p_screen: float = P_SCREEN_DEFAULT,
) -> BatchResult:
    """Process all SPARC galaxies and compute ε statistics.

    Parameters
    ----------
    data_dir : str
        Path to SPARC data directory.
    max_quality : int
        Maximum quality flag (1–3).
    screening : bool
        Apply chameleon screening to all galaxies.

    Returns
    -------
    BatchResult
        Aggregate statistics over the SPARC sample.
    """
    galaxies = load_sparc_database(data_dir, max_quality)
    fits: list[GalaxyFitResult] = []

    for gal in galaxies:
        fit = fit_galaxy(
            gal, eps_min, eps_max, n_steps,
            r_trans_kpc, ml_disk, ml_bulge,
            screening, p_screen,
        )
        fits.append(fit)

    # ── Separate converged fits ──
    converged = [f for f in fits if f.converged]
    improved = [f for f in fits if f.chi2_sdgft < f.chi2_newton]

    # ── ε statistics (converged only) ──
    eps_values = [f.epsilon_best for f in converged]

    if eps_values:
        eps_mean = sum(eps_values) / len(eps_values)
        eps_median = _median(eps_values)
        eps_var = (sum((e - eps_mean) ** 2 for e in eps_values)
                   / max(len(eps_values) - 1, 1))
        eps_std = math.sqrt(eps_var)
        eps_sem = eps_std / math.sqrt(len(eps_values))
    else:
        eps_mean = eps_median = eps_std = eps_sem = 0.0

    # Weight by 1/χ²_red (better fits get more weight)
    weights = []
    w_eps = []
    for f in converged:
        chi2_r = f.chi2_red_sdgft
        if chi2_r > 0:
            weights.append(1.0 / chi2_r)
            w_eps.append(f.epsilon_best)
    eps_w_mean, eps_w_sem = _weighted_mean_sem(w_eps, weights)

    # ── σ from theory ──
    sigma_theory = (abs(eps_mean - EPSILON_THEORY) / eps_sem
                    if eps_sem > 0 else float("inf"))
    sigma_weighted = (abs(eps_w_mean - EPSILON_THEORY) / eps_w_sem
                      if eps_w_sem > 0 else float("inf"))

    # ── χ² medians ──
    chi2_red_n = sorted([f.chi2_red_newton for f in fits])
    chi2_red_s = sorted([f.chi2_red_sdgft for f in fits])
    med_chi2_n = _median(chi2_red_n)
    med_chi2_s = _median(chi2_red_s)

    # ── Quality breakdown ──
    n_q1 = sum(1 for f in fits if f.quality == 1)
    n_q2 = sum(1 for f in fits if f.quality == 2)
    n_q3 = sum(1 for f in fits if f.quality == 3)

    # Per-quality ε
    q1_conv = [f for f in converged if f.quality == 1]
    q2_conv = [f for f in converged if f.quality == 2]
    eps_q1 = (sum(f.epsilon_best for f in q1_conv) / len(q1_conv)
              if q1_conv else 0.0)
    eps_q2 = (sum(f.epsilon_best for f in q2_conv) / len(q2_conv)
              if q2_conv else 0.0)

    return BatchResult(
        fits=tuple(fits),
        n_total=len(galaxies),
        n_fitted=len(fits),
        n_converged=len(converged),
        n_improved=len(improved),
        epsilon_mean=eps_mean,
        epsilon_median=eps_median,
        epsilon_std=eps_std,
        epsilon_sem=eps_sem,
        epsilon_weighted_mean=eps_w_mean,
        epsilon_weighted_sem=eps_w_sem,
        epsilon_theory=EPSILON_THEORY,
        sigma_from_theory=sigma_theory,
        sigma_weighted=sigma_weighted,
        median_chi2_red_newton=med_chi2_n,
        median_chi2_red_sdgft=med_chi2_s,
        n_quality_1=n_q1,
        n_quality_2=n_q2,
        n_quality_3=n_q3,
        epsilon_mean_q1=eps_q1,
        epsilon_mean_q2=eps_q2,
        n_q1_converged=len(q1_conv),
        n_q2_converged=len(q2_conv),
        ml_scan=(),  # populated lazily by run_ml_scan()
    )


# ── M/L sensitivity scan ─────────────────────────────────────────

def scan_ml(
    galaxies: list[SPARCGalaxy],
    ml_values: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    eps_min: float = EPSILON_MIN,
    eps_max: float = EPSILON_MAX,
    n_steps: int = 200,
    r_trans_kpc: float = R_TRANS_KPC,
    screening: bool = True,
    p_screen: float = P_SCREEN_DEFAULT,
) -> list[tuple[float, float, float]]:
    """Scan M/L and compute ⟨ε⟩ for each value.

    Returns list of (M/L, mean_epsilon, SEM) tuples.
    """
    results: list[tuple[float, float, float]] = []

    for ml in ml_values:
        eps_list: list[float] = []
        for gal in galaxies:
            fit = fit_galaxy(
                gal, eps_min, eps_max, n_steps,
                r_trans_kpc, ml, ml + 0.2, screening, p_screen,
            )
            if fit.converged:
                eps_list.append(fit.epsilon_best)

        if eps_list:
            mean_e = sum(eps_list) / len(eps_list)
            var_e = (sum((e - mean_e) ** 2 for e in eps_list)
                     / max(len(eps_list) - 1, 1))
            sem_e = math.sqrt(var_e) / math.sqrt(len(eps_list))
        else:
            mean_e = 0.0
            sem_e = 0.0
        results.append((ml, mean_e, sem_e))

    return results


# ── Module-level computation ──────────────────────────────────────

def _compute_batch() -> BatchResult | None:
    """Attempt to load SPARC and run batch; return None if data missing."""
    try:
        return run_batch()
    except (FileNotFoundError, OSError):
        return None


SPARC_BATCH: BatchResult | None = _compute_batch()


# ── Power-law fitting ─────────────────────────────────────────────

def fit_galaxy_powerlaw(
    galaxy: SPARCGalaxy,
    nu_min: float = NU_MIN,
    nu_max: float = NU_MAX,
    n_steps: int = N_NU_STEPS,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
    screening: bool = True,
    p_screen: float = P_SCREEN_DEFAULT,
) -> PowerlawFitResult:
    """Fit power-law exponent ν and evaluate zero-parameter prediction.

    For each galaxy, computes:
    1. χ² at fixed ν = 2α_M (zero-parameter SDGFT prediction)
    2. Best-fit ν via scan (one-parameter fit)

    Parameters
    ----------
    galaxy : SPARCGalaxy
        Galaxy to fit.
    nu_min, nu_max : float
        ν scan range.
    n_steps : int
        Number of ν values to scan.
    screening : bool
        Apply chameleon screening.

    Returns
    -------
    PowerlawFitResult
    """
    data = galaxy.data

    sigma_screen = 0.0
    if screening:
        sigma_screen = _estimate_sigma_screen(
            data, r_trans_kpc, ml_disk, ml_bulge)

    # ── Zero-parameter prediction (ν = 2α_M) ──
    chi2_n_zero, chi2_zero = _chi2_galaxy_powerlaw(
        data, NU_THEORY, r_trans_kpc, ml_disk, ml_bulge,
        screening, sigma_screen, p_screen,
    )

    # ── Scan ν for best fit ──
    best_nu = nu_min
    best_chi2 = float("inf")
    chi2_n_stored = 0.0

    step = (nu_max - nu_min) / max(n_steps - 1, 1)
    for i in range(n_steps):
        nu = nu_min + i * step
        chi2_n, chi2_s = _chi2_galaxy_powerlaw(
            data, nu, r_trans_kpc, ml_disk, ml_bulge,
            screening, sigma_screen, p_screen,
        )
        if chi2_s < best_chi2:
            best_chi2 = chi2_s
            best_nu = nu
            chi2_n_stored = chi2_n

    converged = (best_nu > nu_min + step and
                 best_nu < nu_max - step)

    return PowerlawFitResult(
        name=galaxy.name,
        nu_best=best_nu,
        chi2_newton=chi2_n_stored,
        chi2_powerlaw_fit=best_chi2,
        chi2_zero_param=chi2_zero,
        n_data=len(data),
        quality=galaxy.catalog.quality,
        distance_mpc=galaxy.catalog.distance_mpc,
        v_flat_kms=galaxy.catalog.v_flat_kms,
        r_max_kpc=galaxy.r_max_kpc,
        converged=converged,
    )


# ── Power-law batch processing ────────────────────────────────────

def run_batch_powerlaw(
    data_dir: str = SPARC_DATA_DIR,
    max_quality: int = MAX_QUALITY,
    nu_min: float = NU_MIN,
    nu_max: float = NU_MAX,
    n_steps: int = N_NU_STEPS,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
    screening: bool = True,
    p_screen: float = P_SCREEN_DEFAULT,
) -> PowerlawBatchResult:
    """Process all SPARC galaxies with power-law G_eff.

    For each galaxy, computes both the zero-parameter prediction
    (ν = 2α_M) and the best-fit ν.  Aggregate statistics quantify
    how close ⟨ν⟩ is to the theoretical 2α_M = 19/43.

    Returns
    -------
    PowerlawBatchResult
    """
    galaxies = load_sparc_database(data_dir, max_quality)
    fits: list[PowerlawFitResult] = []

    for gal in galaxies:
        fit = fit_galaxy_powerlaw(
            gal, nu_min, nu_max, n_steps,
            r_trans_kpc, ml_disk, ml_bulge,
            screening, p_screen,
        )
        fits.append(fit)

    # ── Separate converged fits ──
    converged = [f for f in fits if f.converged]
    improved_fit = [f for f in fits if f.chi2_powerlaw_fit < f.chi2_newton]
    improved_zero = [f for f in fits if f.chi2_zero_param < f.chi2_newton]

    # ── ν statistics (converged only) ──
    nu_values = [f.nu_best for f in converged]
    if nu_values:
        nu_mean = sum(nu_values) / len(nu_values)
        nu_median = _median(nu_values)
        nu_var = (sum((n - nu_mean) ** 2 for n in nu_values)
                  / max(len(nu_values) - 1, 1))
        nu_std = math.sqrt(nu_var)
        nu_sem = nu_std / math.sqrt(len(nu_values))
    else:
        nu_mean = nu_median = nu_std = nu_sem = 0.0

    # Weight by 1/χ²_red
    weights = []
    w_nu = []
    for f in converged:
        chi2_r = f.chi2_red_powerlaw_fit
        if chi2_r > 0:
            weights.append(1.0 / chi2_r)
            w_nu.append(f.nu_best)
    nu_w_mean, nu_w_sem = _weighted_mean_sem(w_nu, weights)

    # σ from theory
    sigma_theory = (abs(nu_mean - NU_THEORY) / nu_sem
                    if nu_sem > 0 else float("inf"))
    sigma_w = (abs(nu_w_mean - NU_THEORY) / nu_w_sem
               if nu_w_sem > 0 else float("inf"))

    # χ² medians
    med_n = _median(sorted([f.chi2_red_newton for f in fits]))
    med_fit = _median(sorted([f.chi2_red_powerlaw_fit for f in fits]))
    med_zero = _median(sorted([f.chi2_red_zero_param for f in fits]))

    return PowerlawBatchResult(
        fits=tuple(fits),
        n_total=len(galaxies),
        n_converged=len(converged),
        n_improved_fit=len(improved_fit),
        n_improved_zero=len(improved_zero),
        nu_mean=nu_mean,
        nu_median=nu_median,
        nu_std=nu_std,
        nu_sem=nu_sem,
        nu_weighted_mean=nu_w_mean,
        nu_weighted_sem=nu_w_sem,
        nu_theory=NU_THEORY,
        sigma_from_theory=sigma_theory,
        sigma_weighted=sigma_w,
        median_chi2_red_newton=med_n,
        median_chi2_red_powerlaw_fit=med_fit,
        median_chi2_red_powerlaw_zero=med_zero,
    )


def _compute_batch_powerlaw() -> PowerlawBatchResult | None:
    """Attempt power-law batch; return None if data missing."""
    try:
        return run_batch_powerlaw()
    except (FileNotFoundError, OSError):
        return None


SPARC_POWERLAW: PowerlawBatchResult | None = _compute_batch_powerlaw()


# ── Dynamic Chameleon Screening ───────────────────────────────────
#
# Standard screening uses Σ_screen = Σ(r_trans) — the same
# "absolute" surface density threshold for every galaxy.  This is
# inconsistent with f(R)-type chameleon physics, where the screening
# depth depends on the **local Newtonian potential** Φ_N.
#
# In the thin-disk limit:  Φ_N ∝ Σ_0 · R_d,  where Σ_0 is the
# central surface mass density and R_d the disk scale length.
# A diffuse dwarf (low Σ_0) is therefore *less* screened than a
# massive spiral (high Σ_0).
#
# Dynamic screening:
#     Σ_screen(galaxy) = f_screen · Σ_0(galaxy)
#
# where f_screen is a *universal* fraction (same for all galaxies)
# and Σ_0 = Υ_disk · SB_disk,0 (central disk surface brightness
# from the SPARC catalog, converted to mass units).
#
# The key conceptual shift: we are NOT adding a free parameter per
# galaxy.  f_screen is determined a priori from the transition
# scale: at r = r_trans the screening should be ~50 %, so
# f_screen ≈ 1  (i.e. Σ_screen ≈ Σ_0).  We explore a few fixed
# values to validate this physically.

#: Dynamic screening fraction.
#  f_screen = 1.0 means the chameleon threshold equals the central
#  density — i.e. screening turns off where Σ(R) ≪ Σ_0.
F_SCREEN_DEFAULT: float = 1.0

#: Minimum Σ_screen to avoid numerical issues [M☉/kpc²].
SIGMA_SCREEN_FLOOR: float = 1e4


def _central_surface_density(
    galaxy: SPARCGalaxy,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """Central surface mass density Σ₀ of a galaxy [M☉/kpc²].

    Uses the catalog's central disk SB (in L☉/pc²) converted by M/L.
    Falls back to the innermost rotation-curve data point if the
    catalog SB is zero or missing.

    Parameters
    ----------
    galaxy : SPARCGalaxy
        Galaxy with catalog and rotation data.

    Returns
    -------
    float
        Σ₀ in M☉/kpc² (>= SIGMA_SCREEN_FLOOR).
    """
    sb0_disk = galaxy.catalog.sb_disk   # L☉/pc²
    sb0_eff = galaxy.catalog.sb_eff     # effective SB

    # Use catalog central SB if available
    if sb0_disk > 0:
        sigma0 = ml_disk * sb0_disk * 1e6  # L☉/pc² → M☉/kpc²
        return max(sigma0, SIGMA_SCREEN_FLOOR)

    # Fallback: effective SB
    if sb0_eff > 0:
        sigma0 = ml_disk * sb0_eff * 1e6
        return max(sigma0, SIGMA_SCREEN_FLOOR)

    # Fallback: innermost data point
    if galaxy.data:
        dp0 = galaxy.data[0]
        sigma0 = _estimate_surface_density(dp0, ml_disk, ml_bulge)
        return max(sigma0, SIGMA_SCREEN_FLOOR)

    return SIGMA_SCREEN_FLOOR


def _dynamic_sigma_screen(
    galaxy: SPARCGalaxy,
    f_screen: float = F_SCREEN_DEFAULT,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
) -> float:
    """Dynamic screening threshold: Σ_screen = f_screen · Σ₀.

    Parameters
    ----------
    galaxy : SPARCGalaxy
        Galaxy to compute screening for.
    f_screen : float
        Universal screening fraction (same for all galaxies).

    Returns
    -------
    float
        Σ_screen in M☉/kpc².
    """
    sigma0 = _central_surface_density(galaxy, ml_disk, ml_bulge)
    return max(f_screen * sigma0, SIGMA_SCREEN_FLOOR)


def fit_galaxy_powerlaw_dynamic(
    galaxy: SPARCGalaxy,
    nu_min: float = NU_MIN,
    nu_max: float = NU_MAX,
    n_steps: int = N_NU_STEPS,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
    f_screen: float = F_SCREEN_DEFAULT,
    p_screen: float = P_SCREEN_DEFAULT,
) -> PowerlawFitResult:
    """Fit ν with dynamic chameleon screening.

    Same as fit_galaxy_powerlaw(), but Σ_screen is set proportionally
    to the galaxy's central surface density rather than interpolated
    at r_trans.

    Parameters
    ----------
    galaxy : SPARCGalaxy
        Galaxy to fit.
    f_screen : float
        Universal screening fraction.

    Returns
    -------
    PowerlawFitResult
    """
    data = galaxy.data
    sigma_screen = _dynamic_sigma_screen(galaxy, f_screen, ml_disk, ml_bulge)

    # ── Zero-parameter prediction (ν = 2α_M) ──
    chi2_n_zero, chi2_zero = _chi2_galaxy_powerlaw(
        data, NU_THEORY, r_trans_kpc, ml_disk, ml_bulge,
        True, sigma_screen, p_screen,
    )

    # ── Scan ν for best fit ──
    best_nu = nu_min
    best_chi2 = float("inf")
    chi2_n_stored = 0.0

    step = (nu_max - nu_min) / max(n_steps - 1, 1)
    for i in range(n_steps):
        nu = nu_min + i * step
        chi2_n, chi2_s = _chi2_galaxy_powerlaw(
            data, nu, r_trans_kpc, ml_disk, ml_bulge,
            True, sigma_screen, p_screen,
        )
        if chi2_s < best_chi2:
            best_chi2 = chi2_s
            best_nu = nu
            chi2_n_stored = chi2_n

    converged = (best_nu > nu_min + step and
                 best_nu < nu_max - step)

    return PowerlawFitResult(
        name=galaxy.name,
        nu_best=best_nu,
        chi2_newton=chi2_n_stored,
        chi2_powerlaw_fit=best_chi2,
        chi2_zero_param=chi2_zero,
        n_data=len(data),
        quality=galaxy.catalog.quality,
        distance_mpc=galaxy.catalog.distance_mpc,
        v_flat_kms=galaxy.catalog.v_flat_kms,
        r_max_kpc=galaxy.r_max_kpc,
        converged=converged,
    )


def run_batch_powerlaw_dynamic(
    data_dir: str = SPARC_DATA_DIR,
    max_quality: int = MAX_QUALITY,
    nu_min: float = NU_MIN,
    nu_max: float = NU_MAX,
    n_steps: int = N_NU_STEPS,
    r_trans_kpc: float = R_TRANS_KPC,
    ml_disk: float = ML_DISK,
    ml_bulge: float = ML_BULGE,
    f_screen: float = F_SCREEN_DEFAULT,
    p_screen: float = P_SCREEN_DEFAULT,
) -> PowerlawBatchResult:
    """Power-law batch with dynamic chameleon screening.

    Σ_screen is set per-galaxy as f_screen × Σ₀ (central surface
    density from the SPARC catalog).  This causes:

    - Dense spirals   → high Σ_screen → strong screening (G ≈ G_N)
    - Diffuse dwarfs  → low Σ_screen  → weak screening  (G ≈ G_eff)

    All other aspects identical to run_batch_powerlaw().

    Returns
    -------
    PowerlawBatchResult
    """
    galaxies = load_sparc_database(data_dir, max_quality)
    fits: list[PowerlawFitResult] = []

    for gal in galaxies:
        fit = fit_galaxy_powerlaw_dynamic(
            gal, nu_min, nu_max, n_steps,
            r_trans_kpc, ml_disk, ml_bulge,
            f_screen, p_screen,
        )
        fits.append(fit)

    # ── Statistics (same code as run_batch_powerlaw) ──
    converged = [f for f in fits if f.converged]
    improved_fit = [f for f in fits if f.chi2_powerlaw_fit < f.chi2_newton]
    improved_zero = [f for f in fits if f.chi2_zero_param < f.chi2_newton]

    nu_values = [f.nu_best for f in converged]
    if nu_values:
        nu_mean = sum(nu_values) / len(nu_values)
        nu_median = _median(nu_values)
        nu_var = (sum((n - nu_mean) ** 2 for n in nu_values)
                  / max(len(nu_values) - 1, 1))
        nu_std = math.sqrt(nu_var)
        nu_sem = nu_std / math.sqrt(len(nu_values))
    else:
        nu_mean = nu_median = nu_std = nu_sem = 0.0

    weights = []
    w_nu = []
    for f in converged:
        chi2_r = f.chi2_red_powerlaw_fit
        if chi2_r > 0:
            weights.append(1.0 / chi2_r)
            w_nu.append(f.nu_best)
    nu_w_mean, nu_w_sem = _weighted_mean_sem(w_nu, weights)

    sigma_theory = (abs(nu_mean - NU_THEORY) / nu_sem
                    if nu_sem > 0 else float("inf"))
    sigma_w = (abs(nu_w_mean - NU_THEORY) / nu_w_sem
               if nu_w_sem > 0 else float("inf"))

    med_n = _median(sorted([f.chi2_red_newton for f in fits]))
    med_fit = _median(sorted([f.chi2_red_powerlaw_fit for f in fits]))
    med_zero = _median(sorted([f.chi2_red_zero_param for f in fits]))

    return PowerlawBatchResult(
        fits=tuple(fits),
        n_total=len(galaxies),
        n_converged=len(converged),
        n_improved_fit=len(improved_fit),
        n_improved_zero=len(improved_zero),
        nu_mean=nu_mean,
        nu_median=nu_median,
        nu_std=nu_std,
        nu_sem=nu_sem,
        nu_weighted_mean=nu_w_mean,
        nu_weighted_sem=nu_w_sem,
        nu_theory=NU_THEORY,
        sigma_from_theory=sigma_theory,
        sigma_weighted=sigma_w,
        median_chi2_red_newton=med_n,
        median_chi2_red_powerlaw_fit=med_fit,
        median_chi2_red_powerlaw_zero=med_zero,
    )


def _compute_batch_dynamic() -> PowerlawBatchResult | None:
    """Attempt dynamic-screening power-law batch."""
    try:
        return run_batch_powerlaw_dynamic()
    except (FileNotFoundError, OSError):
        return None


SPARC_DYNAMIC: PowerlawBatchResult | None = _compute_batch_dynamic()
"""Pre-computed power-law batch with dynamic chameleon screening."""


def print_dynamic_summary(
    dyn: PowerlawBatchResult | None = None,
    static: PowerlawBatchResult | None = None,
    log_batch: BatchResult | None = None,
) -> None:
    """Print dynamic-screening results with comparison to all modes."""
    if dyn is None:
        dyn = SPARC_DYNAMIC
    if dyn is None:
        print("Dynamic screening results not available.")
        return
    if static is None:
        static = SPARC_POWERLAW
    if log_batch is None:
        log_batch = SPARC_BATCH

    print("=" * 80)
    print("  DYNAMIC CHAMELEON SCREENING: Σ_screen = f · Σ₀(galaxy)")
    print(f"  f_screen = {F_SCREEN_DEFAULT:.2f}  |  "
          f"ν = 2α_M = {NU_THEORY:.4f}")
    print("=" * 80)

    pct_fit = 100 * dyn.n_improved_fit / max(dyn.n_total, 1)
    pct_zero = 100 * dyn.n_improved_zero / max(dyn.n_total, 1)
    print(f"\n  Sample: {dyn.n_total} galaxies, "
          f"{dyn.n_converged} converged")
    print(f"    ν-fitted > Newton:  {dyn.n_improved_fit} ({pct_fit:.0f}%)")
    print(f"    Zero-param > Newton: {dyn.n_improved_zero} ({pct_zero:.0f}%)")

    print(f"\n  ν statistics (converged, N={dyn.n_converged}):")
    print(f"    ⟨ν⟩ (unweighted):  {dyn.nu_mean:.4f} ± {dyn.nu_sem:.4f}")
    print(f"    ⟨ν⟩ (weighted):    {dyn.nu_weighted_mean:.4f}"
          f" ± {dyn.nu_weighted_sem:.4f}")
    print(f"    Median ν:           {dyn.nu_median:.4f}")
    print(f"    σ(ν):               {dyn.nu_std:.4f}")

    print(f"\n  Theory: ν = 2α_M = {dyn.nu_theory:.4f}")
    print(f"    |⟨ν⟩ − ν_theory| / SEM = "
          f"{dyn.sigma_from_theory:.2f}σ  (unweighted)")
    print(f"    |⟨ν⟩_w − ν_theory| / SEM_w = "
          f"{dyn.sigma_weighted:.2f}σ  (weighted)")

    # ── Comparison across all four modes ──
    print(f"\n  ═══ COMPARISON: ALL FOUR MODES ═══")
    print(f"  {'Mode':<28} {'⟨param⟩_w':>10} {'σ_theory':>10}"
          f" {'med χ²_red':>10}")
    print("  " + "-" * 62)

    if log_batch:
        print(f"  {'Log (ε fitted)':<28} "
              f"{log_batch.epsilon_weighted_mean:>10.4f} "
              f"{log_batch.sigma_weighted:>10.2f}σ "
              f"{log_batch.median_chi2_red_sdgft:>10.1f}")
    if static:
        print(f"  {'Powerlaw (static screen)':<28} "
              f"{static.nu_weighted_mean:>10.4f} "
              f"{static.sigma_weighted:>10.2f}σ "
              f"{static.median_chi2_red_powerlaw_fit:>10.1f}")
    print(f"  {'Powerlaw (dynamic screen)':<28} "
          f"{dyn.nu_weighted_mean:>10.4f} "
          f"{dyn.sigma_weighted:>10.2f}σ "
          f"{dyn.median_chi2_red_powerlaw_fit:>10.1f}")

    # Zero-param comparison
    print(f"\n  Zero-parameter (ν = 2α_M):")
    if static:
        print(f"    Static screen:  med χ²_red = "
              f"{static.median_chi2_red_powerlaw_zero:.1f}  "
              f"({static.n_improved_zero}/{static.n_total} > Newton)")
    print(f"    Dynamic screen: med χ²_red = "
          f"{dyn.median_chi2_red_powerlaw_zero:.1f}  "
          f"({dyn.n_improved_zero}/{dyn.n_total} > Newton)")

    # ── LSB vs HSB comparison ──
    # Split at median V_flat to separate dwarfs from spirals
    conv = [f for f in dyn.fits if f.converged]
    if conv:
        v_median = _median(sorted([f.v_flat_kms for f in conv]))
        lsb = [f for f in conv if f.v_flat_kms < v_median]
        hsb = [f for f in conv if f.v_flat_kms >= v_median]

        if lsb and hsb:
            nu_lsb = sum(f.nu_best for f in lsb) / len(lsb)
            nu_hsb = sum(f.nu_best for f in hsb) / len(hsb)

            # Also for static
            lsb_s = hsb_s = None
            if static:
                conv_s = [f for f in static.fits if f.converged]
                v_med_s = _median(sorted([f.v_flat_kms for f in conv_s]))
                lsb_s = [f for f in conv_s if f.v_flat_kms < v_med_s]
                hsb_s = [f for f in conv_s if f.v_flat_kms >= v_med_s]

            print(f"\n  ── LSB vs HSB split (V_flat median = "
                  f"{v_median:.0f} km/s) ──")
            print(f"  {'Subset':<18} {'Dynamic ⟨ν⟩':>12}"
                  f" {'Static ⟨ν⟩':>12} {'Theory':>8}")
            print("  " + "-" * 54)

            nu_lsb_s = (sum(f.nu_best for f in lsb_s) / len(lsb_s)
                        if lsb_s else 0.0)
            nu_hsb_s = (sum(f.nu_best for f in hsb_s) / len(hsb_s)
                        if hsb_s else 0.0)

            print(f"  {'LSB (dwarfs)':<18} {nu_lsb:>12.4f}"
                  f" {nu_lsb_s:>12.4f} {NU_THEORY:>8.4f}")
            print(f"  {'HSB (spirals)':<18} {nu_hsb:>12.4f}"
                  f" {nu_hsb_s:>12.4f} {NU_THEORY:>8.4f}")
            print(f"  {'Δν (LSB−HSB)':<18} "
                  f"{nu_lsb - nu_hsb:>+12.4f}"
                  f" {nu_lsb_s - nu_hsb_s:>+12.4f} {'0.0000':>8}")

    print("=" * 80)

def print_summary(batch: BatchResult | None = None,
                   ml_scan: bool = False) -> None:
    """Print formatted SPARC batch analysis results."""
    if batch is None:
        batch = SPARC_BATCH
    if batch is None:
        print("SPARC data not available.")
        return

    b = batch
    print("=" * 80)
    print("  SDGFT SPARC Batch: ε Universality Test (175 Galaxies)")
    print("=" * 80)

    print(f"\n  Sample: {b.n_total} galaxies loaded"
          f"  ({b.n_quality_1} Q=1, {b.n_quality_2} Q=2,"
          f" {b.n_quality_3} Q=3)")
    print(f"    Fitted:    {b.n_fitted}")
    print(f"    Converged: {b.n_converged}"
          f"  ({b.n_converged}/{b.n_fitted}"
          f" = {100*b.n_converged/max(b.n_fitted,1):.0f}%)")
    print(f"    SDGFT > Newton: {b.n_improved}"
          f"  ({100*b.n_improved/max(b.n_fitted,1):.0f}%)")

    print(f"\n  ε statistics (converged, N={b.n_converged}):")
    print(f"    ⟨ε⟩ (unweighted):  {b.epsilon_mean:.4f} ± {b.epsilon_sem:.4f}")
    print(f"    ⟨ε⟩ (weighted):    {b.epsilon_weighted_mean:.4f}"
          f" ± {b.epsilon_weighted_sem:.4f}")
    print(f"    Median ε:           {b.epsilon_median:.4f}")
    print(f"    σ(ε):               {b.epsilon_std:.4f}")

    print(f"\n  Comparison to theory α_M(1−α_M) = {b.epsilon_theory:.4f}:")
    print(f"    |⟨ε⟩ − ε_theory| / SEM = {b.sigma_from_theory:.2f}σ"
          f"  (unweighted)")
    print(f"    |⟨ε⟩_w − ε_theory| / SEM_w = {b.sigma_weighted:.2f}σ"
          f"  (weighted)")

    print(f"\n  χ² comparison (median over sample):")
    print(f"    Newton:  χ²_red = {b.median_chi2_red_newton:.1f}")
    print(f"    SDGFT:   χ²_red = {b.median_chi2_red_sdgft:.1f}")

    # Quality breakdown
    if b.n_q1_converged > 0:
        print(f"\n  By quality (converged only):")
        print(f"    Q=1 (high):   ⟨ε⟩ = {b.epsilon_mean_q1:.4f}"
              f"  (N={b.n_q1_converged})")
    if b.n_q2_converged > 0:
        print(f"    Q=2 (medium): ⟨ε⟩ = {b.epsilon_mean_q2:.4f}"
              f"  (N={b.n_q2_converged})")

    # Top 10 best fits (lowest χ²_red)
    sorted_fits = sorted(b.fits, key=lambda f: f.chi2_red_sdgft)
    print(f"\n  Top 15 galaxies (lowest χ²_red):")
    print(f"  {'Galaxy':<14} {'ε_fit':>7} {'χ²_red':>7} {'χ²_N':>7}"
          f" {'N':>4} {'Q':>2} {'Conv':>5}")
    print("  " + "-" * 50)
    for f in sorted_fits[:15]:
        conv = "✓" if f.converged else "✗"
        print(f"  {f.name:<14} {f.epsilon_best:>7.3f}"
              f" {f.chi2_red_sdgft:>7.1f} {f.chi2_red_newton:>7.1f}"
              f" {f.n_data:>4d} {f.quality:>2d} {conv:>5}")

    # Histogram-like ε distribution
    print(f"\n  ε distribution (converged, N={b.n_converged}):")
    bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
            (0.20, 0.30), (0.30, 0.50), (0.50, 1.0), (1.0, 2.0)]
    conv_fits = [f for f in b.fits if f.converged]
    for lo, hi in bins:
        count = sum(1 for f in conv_fits if lo <= f.epsilon_best < hi)
        bar = "█" * count
        if count > 0:
            print(f"    [{lo:.2f}, {hi:.2f}): {count:>3d} {bar}")

    # M/L sensitivity
    ml_data = b.ml_scan
    if ml_scan and not ml_data:
        print(f"\n  Computing M/L sensitivity scan ...")
        galaxies = load_sparc_database()
        ml_data = tuple(scan_ml(galaxies))

    if ml_data:
        print(f"\n  M/L sensitivity (Υ_disk scan):")
        print(f"  {'Υ_disk':>7} {'⟨ε⟩':>8} {'SEM':>8}"
              f" {'dist to 0.172':>14}")
        print("  " + "-" * 40)
        best_dist = min(abs(e - EPSILON_THEORY) for _, e, _ in ml_data)
        for ml, eps_m, sem_m in ml_data:
            dist = abs(eps_m - EPSILON_THEORY)
            marker = " ←" if abs(dist - best_dist) < 1e-6 else ""
            print(f"  {ml:>7.2f} {eps_m:>8.4f} {sem_m:>8.4f}"
                  f" {dist:>14.4f}{marker}")

    # Key result
    print(f"\n  ═══ KEY RESULT ═══")
    print(f"  ⟨ε⟩_SPARC = {b.epsilon_weighted_mean:.4f}"
          f" ± {b.epsilon_weighted_sem:.4f}"
          f"  (Υ_disk = {ML_DISK})")
    print(f"  ε_theory  = {b.epsilon_theory:.4f}")
    print(f"  Deviation = {b.sigma_weighted:.2f}σ")

    print("=" * 80)


# ── Power-law summary ─────────────────────────────────────────────

def print_powerlaw_summary(
    pl: PowerlawBatchResult | None = None,
    log_batch: BatchResult | None = None,
) -> None:
    """Print power-law SPARC batch results with comparison to log mode."""
    if pl is None:
        pl = SPARC_POWERLAW
    if pl is None:
        print("Power-law SPARC results not available.")
        return
    if log_batch is None:
        log_batch = SPARC_BATCH

    print("=" * 80)
    print("  SDGFT Power-Law Resummation: G_eff(R) = G_N·(R/r_t)^ν")
    print(f"  ν = 2α_M = 19/43 ≈ {NU_THEORY:.4f}  (zero free parameters)")
    print("=" * 80)

    # Sample
    pct_fit = 100 * pl.n_improved_fit / max(pl.n_total, 1)
    pct_zero = 100 * pl.n_improved_zero / max(pl.n_total, 1)
    print(f"\n  Sample: {pl.n_total} galaxies")
    print(f"    Converged (ν fitted):      {pl.n_converged}")
    print(f"    Powerlaw(fit) > Newton:    {pl.n_improved_fit}"
          f" ({pct_fit:.0f}%)")
    print(f"    Powerlaw(ν=2α_M) > Newton: {pl.n_improved_zero}"
          f" ({pct_zero:.0f}%)")

    # ν statistics
    print(f"\n  ν statistics (converged, N={pl.n_converged}):")
    print(f"    ⟨ν⟩ (unweighted):  {pl.nu_mean:.4f} ± {pl.nu_sem:.4f}")
    print(f"    ⟨ν⟩ (weighted):    {pl.nu_weighted_mean:.4f}"
          f" ± {pl.nu_weighted_sem:.4f}")
    print(f"    Median ν:           {pl.nu_median:.4f}")
    print(f"    σ(ν):               {pl.nu_std:.4f}")

    # Theory comparison
    print(f"\n  Theory: ν = 2α_M = {pl.nu_theory:.4f}")
    print(f"    |⟨ν⟩ − ν_theory| / SEM = "
          f"{pl.sigma_from_theory:.2f}σ  (unweighted)")
    print(f"    |⟨ν⟩_w − ν_theory| / SEM_w = "
          f"{pl.sigma_weighted:.2f}σ  (weighted)")

    # χ² comparison across all modes
    print(f"\n  Median χ²_red comparison:")
    print(f"    Newton (baryonic):         {pl.median_chi2_red_newton:.1f}")
    if log_batch:
        print(f"    SDGFT log (ε fitted):      "
              f"{log_batch.median_chi2_red_sdgft:.1f}")
    print(f"    SDGFT powerlaw (ν fitted): "
          f"{pl.median_chi2_red_powerlaw_fit:.1f}")
    print(f"    SDGFT powerlaw (ν=2α_M):   "
          f"{pl.median_chi2_red_powerlaw_zero:.1f}")

    # Top 15 by zero-param chi2_red
    sorted_fits = sorted(pl.fits, key=lambda f: f.chi2_red_zero_param)
    print(f"\n  Top 15 galaxies (lowest χ²_red at ν=2α_M):")
    print(f"  {'Galaxy':<14} {'ν_fit':>6} {'χ²(0p)':>7} {'χ²(fit)':>8}"
          f" {'χ²_N':>7} {'N':>4} {'Q':>2}")
    print("  " + "-" * 54)
    for f in sorted_fits[:15]:
        print(f"  {f.name:<14} {f.nu_best:>6.3f}"
              f" {f.chi2_red_zero_param:>7.1f}"
              f" {f.chi2_red_powerlaw_fit:>8.1f}"
              f" {f.chi2_red_newton:>7.1f}"
              f" {f.n_data:>4d} {f.quality:>2d}")

    # ν distribution histogram
    print(f"\n  ν distribution (converged, N={pl.n_converged}):")
    bins = [(0, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65),
            (0.65, 0.80), (0.80, 1.0), (1.0, 1.5), (1.5, 2.0)]
    conv = [f for f in pl.fits if f.converged]
    for lo, hi in bins:
        count = sum(1 for f in conv if lo <= f.nu_best < hi)
        bar = "█" * count
        marker = " ← 2α_M" if lo <= NU_THEORY < hi else ""
        if count > 0 or marker:
            print(f"    [{lo:.2f}, {hi:.2f}): {count:>3d} {bar}{marker}")

    # Key result
    print(f"\n  ═══ KEY RESULT ═══")
    print(f"  Zero-parameter prediction (ν = 2α_M = {NU_THEORY:.4f}):")
    print(f"    Median χ²_red = {pl.median_chi2_red_powerlaw_zero:.1f}")
    print(f"    {pl.n_improved_zero}/{pl.n_total} galaxies improved"
          f" over Newton ({pct_zero:.0f}%)")
    print(f"\n  Fitted ν convergence:")
    print(f"    ⟨ν⟩_SPARC  = {pl.nu_weighted_mean:.4f}"
          f" ± {pl.nu_weighted_sem:.4f}")
    print(f"    ν_theory   = {NU_THEORY:.4f}")
    print(f"    Deviation  = {pl.sigma_weighted:.2f}σ")
    print("=" * 80)


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register SPARC batch observables (log + power-law modes)."""
    if SPARC_BATCH is not None:
        b = SPARC_BATCH
        registry.register(Observable(
            name="exp_epsilon_sparc_mean",
            symbol="epsilon_SPARC_mean",
            formula="Unweighted mean of bestfit epsilon over 175 SPARC galaxies",
            predicted=b.epsilon_mean,
            observed=EPSILON_THEORY,
            observed_uncertainty=b.epsilon_sem,
            unit="dimensionless",
            level=6,
            d_star_variant="tree",
            dependencies=("alpha_m",),
        ))
        registry.register(Observable(
            name="exp_epsilon_sparc_weighted",
            symbol="epsilon_SPARC_weighted",
            formula="1/chi2-weighted mean of bestfit epsilon over SPARC",
            predicted=b.epsilon_weighted_mean,
            observed=EPSILON_THEORY,
            observed_uncertainty=b.epsilon_weighted_sem,
            unit="dimensionless",
            level=6,
            d_star_variant="tree",
            dependencies=("alpha_m",),
        ))

    if SPARC_POWERLAW is not None:
        p = SPARC_POWERLAW
        registry.register(Observable(
            name="exp_nu_sparc_mean",
            symbol="nu_SPARC_mean",
            formula="Unweighted mean of bestfit nu=2*alpha_M over SPARC",
            predicted=p.nu_mean,
            observed=NU_THEORY,
            observed_uncertainty=p.nu_sem,
            unit="dimensionless",
            level=6,
            d_star_variant="tree",
            dependencies=("alpha_m",),
        ))
        registry.register(Observable(
            name="exp_nu_sparc_weighted",
            symbol="nu_SPARC_weighted",
            formula="1/chi2-weighted mean of bestfit nu over SPARC",
            predicted=p.nu_weighted_mean,
            observed=NU_THEORY,
            observed_uncertainty=p.nu_weighted_sem,
            unit="dimensionless",
            level=6,
            d_star_variant="tree",
            dependencies=("alpha_m",),
        ))

    if SPARC_DYNAMIC is not None:
        d = SPARC_DYNAMIC
        registry.register(Observable(
            name="exp_nu_sparc_dynamic_weighted",
            symbol="nu_SPARC_dyn_w",
            formula="1/chi2-weighted mean nu with dynamic chameleon screening",
            predicted=d.nu_weighted_mean,
            observed=NU_THEORY,
            observed_uncertainty=d.nu_weighted_sem,
            unit="dimensionless",
            level=6,
            d_star_variant="tree",
            dependencies=("alpha_m",),
        ))


# ── CLI entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    print_summary(ml_scan=True)
    print()
    print_powerlaw_summary()
    print()
    print_dynamic_summary()

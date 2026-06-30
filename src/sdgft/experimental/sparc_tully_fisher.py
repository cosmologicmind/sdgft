"""SPARC Baryonic Tully-Fisher Relation — SDGFT validation.

Tests the SDGFT prediction  b_TF = D* + 1 = 91/24 ≈ 3.79
against the full SPARC galaxy catalog (Lelli, McGaugh, Schombert 2016).

The Baryonic Tully-Fisher Relation (BTFR)
------------------------------------------
In log-space:

    log₁₀(M_bar / M☉) = b · log₁₀(V_flat / km s⁻¹) + a

SDGFT predicts the slope from purely geometric considerations:

    b_TF = D* + 1 = 67/24 + 1 = 91/24 ≈ 3.7917

Observed (Lelli, McGaugh, Schombert 2019, ApJ 877 6):

    b = 3.85 ± 0.09

Baryonic mass computation from SPARC 3.6 μm photometry:

    M_bar = Υ_disk × L_[3.6] × 10⁹ + 1.33 × M_HI × 10⁹   [M☉]

The factor 1.33 corrects the HI mass for primordial helium.
Default Υ_disk = 0.5 M☉/L☉ at 3.6 μm (Schombert & McGaugh 2014).

Exports
-------
- TFGalaxy           — per-galaxy TF data point
- TFResult           — full BTFR fit result
- build_tf_dataset   — load SPARC, compute M_bar, filter V_flat > 0
- fit_btfr           — orthogonal-distance regression (ODR) for BTFR
- compute_tf_result  — one-shot analysis: load + fit + residuals
- plot_btfr          — publication-quality BTFR plot
- plot_residuals     — residual scatter plot
- plot_nu_vs_vflat   — power-law ν vs V_flat correlation
- plot_combined      — multi-panel figure combining all plots
- print_tf_summary   — formatted text summary
- TF_RESULT          — module-level pre-computed result
- register_all       — register b_TF observable
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from sdgft.gravity import ALPHA_M_TREE_F
from sdgft.tully_fisher import B_TF_TREE_F, R_TRANS_KPC
from sdgft.registry import REGISTRY, Observable

from . import sparc_batch as sb

# ── Constants ─────────────────────────────────────────────────────

#: SDGFT-predicted Tully-Fisher slope.
B_TF_THEORY: float = B_TF_TREE_F  # 91/24 ≈ 3.7917

#: Observed BTFR slope (Lelli, McGaugh, Schombert 2019).
B_TF_OBSERVED: float = 3.85
B_TF_OBSERVED_ERR: float = 0.09

#: Default stellar M/L at 3.6 μm (same as sparc_batch).
ML_DISK: float = sb.ML_DISK
ML_BULGE: float = sb.ML_BULGE

#: Helium correction factor for HI mass → total gas mass.
HELIUM_FACTOR: float = 1.33

#: Minimum V_flat to include in TF analysis (km/s).
#  V_flat = 0 means no measured flat velocity.
VFLAT_MIN: float = 10.0

#: Power-law exponent from sparc_batch.
NU_THEORY: float = sb.NU_THEORY


# ── Data structures ───────────────────────────────────────────────

@dataclass(frozen=True)
class TFGalaxy:
    """One galaxy in the Tully-Fisher analysis."""

    name: str
    v_flat_kms: float
    e_v_flat_kms: float
    log_v_flat: float            # log₁₀(V_flat / km s⁻¹)

    m_bar_msun: float            # Total baryonic mass [M☉]
    m_star_msun: float           # Stellar mass [M☉]
    m_gas_msun: float            # Gas mass (incl. He) [M☉]
    log_m_bar: float             # log₁₀(M_bar / M☉)

    luminosity_1e9Lsun: float    # 3.6 μm luminosity
    m_hi_1e9Msun: float          # HI mass
    distance_mpc: float
    quality: int                 # SPARC quality flag (1/2/3)
    hubble_type: int

    # Optional: power-law fit result (if available)
    nu_fit: float | None = None           # Best-fit ν from sparc_batch
    chi2_red_newton: float | None = None
    chi2_red_powerlaw: float | None = None


@dataclass(frozen=True)
class TFResult:
    """Complete Baryonic Tully-Fisher analysis result."""

    galaxies: tuple[TFGalaxy, ...]
    n_galaxies: int

    # ── ODR fit: log(M_bar) = b_fit · log(V_flat) + a_fit ──
    b_fit: float                 # Fitted slope
    b_fit_err: float             # Standard error on slope
    a_fit: float                 # Fitted intercept
    a_fit_err: float             # Standard error on intercept

    # ── Fixed-slope fit: log(M_bar) = b_TF · log(V_flat) + a_fixed ──
    a_fixed: float               # Intercept when slope = b_TF = 91/24
    a_fixed_err: float

    # ── Scatter ──
    rms_scatter: float           # RMS scatter in log(M_bar) around fit
    rms_scatter_fixed: float     # RMS scatter around fixed-slope line
    intrinsic_scatter: float     # Estimated intrinsic scatter (obs subtracted)

    # ── Theory comparison ──
    b_theory: float              # = 91/24
    sigma_b: float               # |b_fit - b_theory| / b_fit_err

    # ── Quality breakdown ──
    n_quality_1: int
    n_quality_2: int
    n_quality_3: int


# ── Baryonic mass computation ─────────────────────────────────────

def baryonic_mass_msun(
    luminosity_1e9Lsun: float,
    m_hi_1e9Msun: float,
    ml_disk: float = ML_DISK,
) -> tuple[float, float, float]:
    """Compute baryonic mass from SPARC photometry.

    Parameters
    ----------
    luminosity_1e9Lsun : float
        Total 3.6 μm luminosity in 10⁹ L☉.
    m_hi_1e9Msun : float
        Total HI mass in 10⁹ M☉.
    ml_disk : float
        Stellar mass-to-light ratio at 3.6 μm.

    Returns
    -------
    (m_bar, m_star, m_gas) : tuple[float, float, float]
        Total baryonic, stellar, and gas masses in M☉.
    """
    m_star = ml_disk * luminosity_1e9Lsun * 1e9
    m_gas = HELIUM_FACTOR * m_hi_1e9Msun * 1e9
    m_bar = m_star + m_gas
    return m_bar, m_star, m_gas


# ── Dataset building ──────────────────────────────────────────────

def build_tf_dataset(
    ml_disk: float = ML_DISK,
    v_flat_min: float = VFLAT_MIN,
    max_quality: int = 3,
    data_dir: str = sb.SPARC_DATA_DIR,
) -> list[TFGalaxy]:
    """Load SPARC catalog and build Tully-Fisher dataset.

    Excludes galaxies with V_flat < v_flat_min or V_flat = 0.
    Joins with power-law fit results if SPARC_POWERLAW is available.

    Parameters
    ----------
    ml_disk : float
        Stellar mass-to-light ratio.
    v_flat_min : float
        Minimum V_flat in km/s.
    max_quality : int
        Maximum SPARC quality flag (1=high only, 3=all).
    data_dir : str
        Path to SPARC data directory.

    Returns
    -------
    list[TFGalaxy]
        Galaxies with valid V_flat and baryonic mass.
    """
    catalog = sb.load_sparc_catalog(data_dir)

    # Build lookup for power-law fits
    pl_lookup: dict[str, sb.PowerlawFitResult] = {}
    if sb.SPARC_POWERLAW is not None:
        for f in sb.SPARC_POWERLAW.fits:
            pl_lookup[f.name] = f

    galaxies: list[TFGalaxy] = []
    for entry in catalog:
        if entry.quality > max_quality:
            continue
        if entry.v_flat_kms < v_flat_min:
            continue
        if entry.luminosity_1e9Lsun <= 0:
            continue

        m_bar, m_star, m_gas = baryonic_mass_msun(
            entry.luminosity_1e9Lsun, entry.m_hi_1e9Msun, ml_disk,
        )
        if m_bar <= 0:
            continue

        log_v = math.log10(entry.v_flat_kms)
        log_m = math.log10(m_bar)

        # Power-law fit result (if available)
        pf = pl_lookup.get(entry.name)
        nu_fit = pf.nu_best if pf and pf.converged else None
        chi2_n = pf.chi2_red_newton if pf else None
        chi2_p = pf.chi2_red_powerlaw_fit if pf else None

        galaxies.append(TFGalaxy(
            name=entry.name,
            v_flat_kms=entry.v_flat_kms,
            e_v_flat_kms=entry.e_v_flat_kms,
            log_v_flat=log_v,
            m_bar_msun=m_bar,
            m_star_msun=m_star,
            m_gas_msun=m_gas,
            log_m_bar=log_m,
            luminosity_1e9Lsun=entry.luminosity_1e9Lsun,
            m_hi_1e9Msun=entry.m_hi_1e9Msun,
            distance_mpc=entry.distance_mpc,
            quality=entry.quality,
            hubble_type=entry.hubble_type,
            nu_fit=nu_fit,
            chi2_red_newton=chi2_n,
            chi2_red_powerlaw=chi2_p,
        ))

    return sorted(galaxies, key=lambda g: g.name)


# ── BTFR fitting ──────────────────────────────────────────────────

def _fit_odr(
    log_v: list[float],
    log_m: list[float],
    e_log_v: list[float] | None = None,
    e_log_m: list[float] | None = None,
) -> tuple[float, float, float, float]:
    """Orthogonal Distance Regression for log(M) = b·log(V) + a.

    Falls back to least-squares if scipy.odr is unavailable.

    Returns
    -------
    (b, b_err, a, a_err)
    """
    import numpy as np

    x = np.array(log_v)
    y = np.array(log_m)

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning,
                                    module="scipy.odr")
            from scipy.odr import ODR, Model, RealData

        def linear(beta, x):  # noqa: E741
            return beta[0] * x + beta[1]

        model = Model(linear)

        # Use uniform weights if errors not provided
        sx = np.array(e_log_v) if e_log_v else np.ones_like(x) * 0.05
        sy = np.array(e_log_m) if e_log_m else np.ones_like(y) * 0.15

        data = RealData(x, y, sx=1.0 / sx, sy=1.0 / sy)
        odr = ODR(data, model, beta0=[3.8, 2.0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            output = odr.run()

        b, a = output.beta
        b_err, a_err = output.sd_beta
        return float(b), float(b_err), float(a), float(a_err)

    except ImportError:
        # Fallback: simple least-squares
        coeffs = np.polyfit(x, y, 1, cov=True)
        b, a = coeffs[0]
        cov = coeffs[1]
        b_err = float(np.sqrt(cov[0, 0]))
        a_err = float(np.sqrt(cov[1, 1]))
        return float(b), b_err, float(a), a_err


def fit_btfr(galaxies: list[TFGalaxy]) -> TFResult:
    """Fit the Baryonic Tully-Fisher Relation.

    Performs:
    1. Free ODR fit: log(M) = b·log(V) + a
    2. Fixed-slope fit: log(M) = b_TF·log(V) + a_fixed
    3. Scatter estimation
    4. Theory comparison

    Parameters
    ----------
    galaxies : list[TFGalaxy]
        Galaxies with valid V_flat and M_bar.

    Returns
    -------
    TFResult
        Complete analysis result.
    """
    import numpy as np

    n = len(galaxies)
    log_v = [g.log_v_flat for g in galaxies]
    log_m = [g.log_m_bar for g in galaxies]

    # Error estimates:
    # δlog(V) ≈ δV / (V · ln 10)
    # δlog(M) ≈ 0.15 dex (systematic from M/L uncertainty ~0.1 dex + distance)
    e_log_v = [
        g.e_v_flat_kms / (g.v_flat_kms * math.log(10))
        if g.e_v_flat_kms > 0 else 0.05
        for g in galaxies
    ]
    e_log_m = [0.15] * n  # Systematic floor

    # ── Free fit ──
    b_fit, b_err, a_fit, a_err = _fit_odr(log_v, log_m, e_log_v, e_log_m)

    # ── Fixed-slope fit: minimise Σ(log_m - b_TF·log_v - a)² ──
    lv = np.array(log_v)
    lm = np.array(log_m)
    a_fixed = float(np.mean(lm - B_TF_THEORY * lv))
    residuals_fixed = lm - B_TF_THEORY * lv - a_fixed
    a_fixed_err = float(np.std(residuals_fixed) / math.sqrt(n))

    # ── Scatter ──
    residuals_fit = lm - b_fit * lv - a_fit
    rms_scatter = float(np.sqrt(np.mean(residuals_fit ** 2)))
    rms_scatter_fixed = float(np.sqrt(np.mean(residuals_fixed ** 2)))

    # Intrinsic scatter: σ_int² ≈ σ_obs² - ⟨δ²⟩
    mean_err2 = float(np.mean(np.array(e_log_m) ** 2))
    var_resid = float(np.var(residuals_fit))
    intrinsic_var = max(var_resid - mean_err2, 0.0)
    intrinsic_scatter = math.sqrt(intrinsic_var)

    # ── Theory comparison ──
    sigma_b = abs(b_fit - B_TF_THEORY) / b_err if b_err > 0 else float("inf")

    # ── Quality breakdown ──
    n_q1 = sum(1 for g in galaxies if g.quality == 1)
    n_q2 = sum(1 for g in galaxies if g.quality == 2)
    n_q3 = sum(1 for g in galaxies if g.quality == 3)

    return TFResult(
        galaxies=tuple(galaxies),
        n_galaxies=n,
        b_fit=b_fit,
        b_fit_err=b_err,
        a_fit=a_fit,
        a_fit_err=a_err,
        a_fixed=a_fixed,
        a_fixed_err=a_fixed_err,
        rms_scatter=rms_scatter,
        rms_scatter_fixed=rms_scatter_fixed,
        intrinsic_scatter=intrinsic_scatter,
        b_theory=B_TF_THEORY,
        sigma_b=sigma_b,
        n_quality_1=n_q1,
        n_quality_2=n_q2,
        n_quality_3=n_q3,
    )


# ── One-shot analysis ─────────────────────────────────────────────

def compute_tf_result(
    ml_disk: float = ML_DISK,
    v_flat_min: float = VFLAT_MIN,
    max_quality: int = 3,
) -> TFResult | None:
    """Build dataset and fit BTFR in one call.

    Returns None if SPARC data is unavailable.
    """
    try:
        galaxies = build_tf_dataset(ml_disk=ml_disk, v_flat_min=v_flat_min,
                                    max_quality=max_quality)
    except (FileNotFoundError, OSError):
        return None

    if len(galaxies) < 10:
        return None

    return fit_btfr(galaxies)


def _compute_module_result() -> TFResult | None:
    """Module-level computation (runs at import)."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return compute_tf_result()
    except Exception:
        return None


TF_RESULT: TFResult | None = _compute_module_result()
"""Pre-computed Tully-Fisher analysis (or None if data unavailable)."""


# ── Plotting ──────────────────────────────────────────────────────

def plot_btfr(
    result: TFResult | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 7),
) -> object | None:
    """Publication-quality Baryonic Tully-Fisher plot.

    log₁₀(M_bar) vs log₁₀(V_flat) with:
    - Data points coloured by SPARC quality flag
    - SDGFT theoretical line (b = 91/24, orange)
    - ODR best-fit line (blue dashed)
    - Observed relation (Lelli+2019, grey dotted)

    Parameters
    ----------
    result : TFResult
        Analysis result. Uses TF_RESULT if None.
    save_path : str
        If given, save figure to this path (PNG/PDF).
    show : bool
        Whether to call plt.show().
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object, or None if matplotlib unavailable.
    """
    if result is None:
        result = TF_RESULT
    if result is None:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # ── Data points by quality ──
    q_colors = {1: "#2166AC", 2: "#67A9CF", 3: "#D1E5F0"}
    q_labels = {1: "Q = 1 (high)", 2: "Q = 2 (med)", 3: "Q = 3 (low)"}
    q_markers = {1: "o", 2: "s", 3: "D"}
    q_sizes = {1: 28, 2: 22, 3: 16}

    for q in (3, 2, 1):  # Plot low quality first (behind)
        gals = [g for g in result.galaxies if g.quality == q]
        if not gals:
            continue
        lv = [g.log_v_flat for g in gals]
        lm = [g.log_m_bar for g in gals]
        ax.scatter(lv, lm, c=q_colors[q], s=q_sizes[q],
                   marker=q_markers[q], label=q_labels[q],
                   edgecolors="white", linewidths=0.3, zorder=3 + q,
                   alpha=0.85)

    # ── Regression lines ──
    v_range = np.linspace(1.2, 2.7, 200)

    # SDGFT theoretical: b = 91/24
    m_sdgft = B_TF_THEORY * v_range + result.a_fixed
    ax.plot(v_range, m_sdgft, color="#D95F02", lw=2.2, zorder=6,
            label=f"SDGFT: b = 91/24 ≈ {B_TF_THEORY:.3f}")

    # ODR best-fit
    m_fit = result.b_fit * v_range + result.a_fit
    ax.plot(v_range, m_fit, color="#1B9E77", lw=1.8, ls="--", zorder=5,
            label=f"ODR fit: b = {result.b_fit:.3f} ± {result.b_fit_err:.3f}")

    # Lelli+2019 observed (b=3.85, a≈2.0)
    a_lelli = result.a_fixed + (B_TF_OBSERVED - B_TF_THEORY) * np.mean(
        [g.log_v_flat for g in result.galaxies]
    )
    m_lelli = B_TF_OBSERVED * v_range + a_lelli
    ax.plot(v_range, m_lelli, color="grey", lw=1.2, ls=":", zorder=4,
            label=f"Lelli+2019: b = {B_TF_OBSERVED:.2f} ± {B_TF_OBSERVED_ERR:.2f}")

    ax.set_xlabel(r"$\log_{10}(V_{\mathrm{flat}} \,/\, \mathrm{km\,s^{-1}})$",
                  fontsize=13)
    ax.set_ylabel(r"$\log_{10}(M_{\mathrm{bar}} \,/\, M_\odot)$",
                  fontsize=13)
    ax.set_title("Baryonic Tully-Fisher Relation — SPARC × SDGFT",
                 fontsize=14, fontweight="bold")

    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--")
    ax.tick_params(labelsize=11)

    # Annotation box
    txt = (f"N = {result.n_galaxies} galaxies\n"
           f"b_{{SDGFT}} = 91/24 ≈ {B_TF_THEORY:.4f}\n"
           f"b_{{fit}} = {result.b_fit:.3f} ± {result.b_fit_err:.3f}\n"
           f"Δb / σ = {result.sigma_b:.2f}σ\n"
           f"σ_{{scatter}} = {result.rms_scatter:.3f} dex")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat",
                      alpha=0.8))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_residuals(
    result: TFResult | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 4.5),
) -> object | None:
    """Residual plot: Δlog(M_bar) vs log(V_flat).

    Shows deviations from the SDGFT theoretical BTFR (b = 91/24).

    Parameters
    ----------
    result : TFResult
        Analysis result. Uses TF_RESULT if None.
    save_path : str
        If given, save figure.
    show : bool
        Whether to call plt.show().
    """
    if result is None:
        result = TF_RESULT
    if result is None:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Residuals from fixed-slope SDGFT line
    q_colors = {1: "#2166AC", 2: "#67A9CF", 3: "#D1E5F0"}
    q_markers = {1: "o", 2: "s", 3: "D"}
    q_sizes = {1: 28, 2: 22, 3: 16}

    for q in (3, 2, 1):
        gals = [g for g in result.galaxies if g.quality == q]
        if not gals:
            continue
        lv = [g.log_v_flat for g in gals]
        resid = [g.log_m_bar - B_TF_THEORY * g.log_v_flat - result.a_fixed
                 for g in gals]
        ax.scatter(lv, resid, c=q_colors[q], s=q_sizes[q],
                   marker=q_markers[q], edgecolors="white", linewidths=0.3,
                   alpha=0.85, zorder=3 + q)

    v_range = [1.2, 2.7]
    ax.axhline(0, color="#D95F02", lw=1.5, zorder=2,
               label=f"SDGFT: b = 91/24")
    ax.fill_between(v_range, -result.rms_scatter_fixed,
                    result.rms_scatter_fixed,
                    alpha=0.12, color="#D95F02", zorder=1,
                    label=f"±1σ scatter = {result.rms_scatter_fixed:.3f} dex")

    ax.set_xlabel(
        r"$\log_{10}(V_{\mathrm{flat}} \,/\, \mathrm{km\,s^{-1}})$",
        fontsize=13)
    ax.set_ylabel(
        r"$\Delta\log_{10}(M_{\mathrm{bar}})$  "
        r"$[\mathrm{data} - \mathrm{SDGFT}]$", fontsize=13)
    ax.set_title("BTFR Residuals (SDGFT slope 91/24)", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--")
    ax.set_ylim(-1.0, 1.0)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_nu_vs_vflat(
    result: TFResult | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 5),
) -> object | None:
    """Power-law exponent ν vs V_flat — tests universality.

    If ν is truly universal (= 2α_M = 19/43), there should be no
    systematic trend with V_flat.

    Parameters
    ----------
    result : TFResult
        Analysis result. Uses TF_RESULT if None.
    save_path : str
        If given, save figure.
    show : bool
        Whether to call plt.show().
    """
    if result is None:
        result = TF_RESULT
    if result is None:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    # Filter galaxies with power-law fits
    gals_with_nu = [g for g in result.galaxies if g.nu_fit is not None]
    if len(gals_with_nu) < 5:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    v_flat = np.array([g.v_flat_kms for g in gals_with_nu])
    nu = np.array([g.nu_fit for g in gals_with_nu])
    qual = np.array([g.quality for g in gals_with_nu])

    q_colors = {1: "#2166AC", 2: "#67A9CF", 3: "#D1E5F0"}
    q_markers = {1: "o", 2: "s", 3: "D"}

    for q in (3, 2, 1):
        mask = qual == q
        if not np.any(mask):
            continue
        ax.scatter(v_flat[mask], nu[mask], c=q_colors[q], s=25,
                   marker=q_markers[q], edgecolors="white", linewidths=0.3,
                   alpha=0.8, zorder=3 + q)

    # Theory line
    ax.axhline(NU_THEORY, color="#D95F02", lw=2, zorder=6,
               label=f"ν = 2α_M = 19/43 ≈ {NU_THEORY:.4f}")

    # Weighted mean
    if sb.SPARC_POWERLAW is not None:
        nu_w = sb.SPARC_POWERLAW.nu_weighted_mean
        nu_w_err = sb.SPARC_POWERLAW.nu_weighted_sem
        ax.axhline(nu_w, color="#1B9E77", lw=1.5, ls="--", zorder=5,
                   label=f"⟨ν⟩_w = {nu_w:.4f} ± {nu_w_err:.4f}")
        ax.axhspan(nu_w - nu_w_err, nu_w + nu_w_err,
                   alpha=0.1, color="#1B9E77", zorder=1)

    # Linear trend fit
    log_v = np.log10(v_flat)
    coeffs = np.polyfit(log_v, nu, 1)
    v_line = np.linspace(v_flat.min(), v_flat.max(), 100)
    nu_trend = np.polyval(coeffs, np.log10(v_line))
    ax.plot(v_line, nu_trend, color="grey", lw=1, ls=":", alpha=0.7,
            label=f"Trend: dν/dlog(V) = {coeffs[0]:.3f}")

    ax.set_xlabel(r"$V_{\mathrm{flat}}$ [km/s]", fontsize=13)
    ax.set_ylabel(r"Best-fit $\nu$", fontsize=13)
    ax.set_title(r"Power-Law Exponent $\nu$ vs $V_{\mathrm{flat}}$",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--")
    ax.set_ylim(-0.1, 2.1)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_combined(
    result: TFResult | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (16, 12),
) -> object | None:
    """Multi-panel figure: BTFR + residuals + ν correlation + histogram.

    Four panels:
    (a) Main BTFR (log M vs log V)
    (b) BTFR residuals
    (c) ν vs V_flat
    (d) ν histogram with theory line

    Parameters
    ----------
    result : TFResult
        Analysis result. Uses TF_RESULT if None.
    save_path : str
        If given, save figure.
    show : bool
        Whether to call plt.show().
    """
    if result is None:
        result = TF_RESULT
    if result is None:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    q_colors = {1: "#2166AC", 2: "#67A9CF", 3: "#D1E5F0"}
    q_labels = {1: "Q = 1", 2: "Q = 2", 3: "Q = 3"}
    q_markers = {1: "o", 2: "s", 3: "D"}
    q_sizes = {1: 22, 2: 18, 3: 14}

    # ── Panel (a): Main BTFR ──
    ax = axes[0, 0]
    for q in (3, 2, 1):
        gals = [g for g in result.galaxies if g.quality == q]
        if not gals:
            continue
        lv = [g.log_v_flat for g in gals]
        lm = [g.log_m_bar for g in gals]
        ax.scatter(lv, lm, c=q_colors[q], s=q_sizes[q],
                   marker=q_markers[q], label=q_labels[q],
                   edgecolors="white", linewidths=0.3, zorder=3 + q,
                   alpha=0.85)

    v_range = np.linspace(1.2, 2.7, 200)
    m_sdgft = B_TF_THEORY * v_range + result.a_fixed
    m_fit = result.b_fit * v_range + result.a_fit
    ax.plot(v_range, m_sdgft, color="#D95F02", lw=2,
            label=f"SDGFT b = {B_TF_THEORY:.3f}")
    ax.plot(v_range, m_fit, color="#1B9E77", lw=1.5, ls="--",
            label=f"Fit b = {result.b_fit:.3f}")

    ax.set_xlabel(r"$\log_{10}(V_{\mathrm{flat}})$", fontsize=11)
    ax.set_ylabel(r"$\log_{10}(M_{\mathrm{bar}}/M_\odot)$", fontsize=11)
    ax.set_title("(a) Baryonic Tully-Fisher", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--")

    # ── Panel (b): Residuals ──
    ax = axes[0, 1]
    for q in (3, 2, 1):
        gals = [g for g in result.galaxies if g.quality == q]
        if not gals:
            continue
        lv = [g.log_v_flat for g in gals]
        resid = [g.log_m_bar - B_TF_THEORY * g.log_v_flat - result.a_fixed
                 for g in gals]
        ax.scatter(lv, resid, c=q_colors[q], s=q_sizes[q],
                   marker=q_markers[q], edgecolors="white", linewidths=0.3,
                   alpha=0.85, zorder=3 + q)

    ax.axhline(0, color="#D95F02", lw=1.5, zorder=2)
    ax.fill_between([1.2, 2.7], -result.rms_scatter_fixed,
                    result.rms_scatter_fixed,
                    alpha=0.12, color="#D95F02")
    ax.set_xlabel(r"$\log_{10}(V_{\mathrm{flat}})$", fontsize=11)
    ax.set_ylabel(r"$\Delta\log_{10}(M_{\mathrm{bar}})$", fontsize=11)
    ax.set_title(f"(b) Residuals (σ = {result.rms_scatter_fixed:.3f} dex)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")
    ax.set_ylim(-1.0, 1.0)

    # ── Panel (c): ν vs V_flat ──
    ax = axes[1, 0]
    gals_nu = [g for g in result.galaxies if g.nu_fit is not None]
    if gals_nu:
        v_arr = np.array([g.v_flat_kms for g in gals_nu])
        nu_arr = np.array([g.nu_fit for g in gals_nu])
        q_arr = np.array([g.quality for g in gals_nu])

        for q in (3, 2, 1):
            mask = q_arr == q
            if not np.any(mask):
                continue
            ax.scatter(v_arr[mask], nu_arr[mask], c=q_colors[q],
                       s=q_sizes[q], marker=q_markers[q],
                       edgecolors="white", linewidths=0.3, alpha=0.8,
                       zorder=3 + q)

        ax.axhline(NU_THEORY, color="#D95F02", lw=2, zorder=6,
                   label=f"ν = 2α_M = {NU_THEORY:.4f}")

        if sb.SPARC_POWERLAW is not None:
            nu_w = sb.SPARC_POWERLAW.nu_weighted_mean
            ax.axhline(nu_w, color="#1B9E77", lw=1.5, ls="--", zorder=5,
                       label=f"⟨ν⟩_w = {nu_w:.4f}")

        # Trend line
        coeffs = np.polyfit(np.log10(v_arr), nu_arr, 1)
        v_line = np.linspace(v_arr.min(), v_arr.max(), 100)
        ax.plot(v_line, np.polyval(coeffs, np.log10(v_line)),
                color="grey", lw=1, ls=":", alpha=0.7)

        ax.legend(fontsize=8, framealpha=0.9)

    ax.set_xlabel(r"$V_{\mathrm{flat}}$ [km/s]", fontsize=11)
    ax.set_ylabel(r"Best-fit $\nu$", fontsize=11)
    ax.set_title(r"(c) Power-Law $\nu$ vs $V_{\mathrm{flat}}$",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")
    ax.set_ylim(-0.1, 2.1)

    # ── Panel (d): ν histogram ──
    ax = axes[1, 1]
    if gals_nu:
        ax.hist(nu_arr, bins=25, range=(0, 2.0), color="#67A9CF",
                edgecolor="white", alpha=0.8, zorder=2)
        ax.axvline(NU_THEORY, color="#D95F02", lw=2, zorder=6,
                   label=f"ν_theory = {NU_THEORY:.4f}")

        if sb.SPARC_POWERLAW is not None:
            nu_w = sb.SPARC_POWERLAW.nu_weighted_mean
            nu_w_err = sb.SPARC_POWERLAW.nu_weighted_sem
            ax.axvline(nu_w, color="#1B9E77", lw=1.5, ls="--", zorder=5,
                       label=f"⟨ν⟩_w = {nu_w:.4f}")

        ax.legend(fontsize=8, framealpha=0.9)

    ax.set_xlabel(r"$\nu$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(r"(d) Distribution of fitted $\nu$",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")

    fig.suptitle(
        f"SDGFT vs SPARC: b_TF = 91/24 ≈ {B_TF_THEORY:.4f}, "
        f"ν = 2α_M = {NU_THEORY:.4f}  |  "
        f"N = {result.n_galaxies} galaxies",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ── Text summary ──────────────────────────────────────────────────

def print_tf_summary(result: TFResult | None = None) -> None:
    """Print formatted Tully-Fisher analysis summary."""
    if result is None:
        result = TF_RESULT
    if result is None:
        print("  [TF analysis unavailable — no SPARC data]")
        return

    r = result
    print("=" * 72)
    print("BARYONIC TULLY-FISHER RELATION — SPARC × SDGFT")
    print("=" * 72)

    print(f"\n  Dataset: {r.n_galaxies} galaxies "
          f"(Q1={r.n_quality_1}, Q2={r.n_quality_2}, Q3={r.n_quality_3})")
    print(f"  M/L (3.6 μm): Υ_disk = {ML_DISK}, Υ_bulge = {ML_BULGE}")
    print(f"  M_bar = Υ_disk × L_[3.6] + 1.33 × M_HI")

    print(f"\n  ── Free ODR Fit ──")
    print(f"  log₁₀(M_bar) = b × log₁₀(V_flat) + a")
    print(f"    b_fit = {r.b_fit:.4f} ± {r.b_fit_err:.4f}")
    print(f"    a_fit = {r.a_fit:.4f} ± {r.a_fit_err:.4f}")
    print(f"    RMS scatter = {r.rms_scatter:.4f} dex")

    print(f"\n  ── SDGFT Fixed-Slope Fit (b = 91/24) ──")
    print(f"    b_SDGFT = {r.b_theory:.4f}")
    print(f"    a_fixed = {r.a_fixed:.4f} ± {r.a_fixed_err:.4f}")
    print(f"    RMS scatter = {r.rms_scatter_fixed:.4f} dex")
    print(f"    Intrinsic scatter = {r.intrinsic_scatter:.4f} dex")

    print(f"\n  ── Theory Comparison ──")
    print(f"    b_SDGFT         = {r.b_theory:.4f}  (= 91/24)")
    print(f"    b_fit (ODR)     = {r.b_fit:.4f} ± {r.b_fit_err:.4f}")
    print(f"    b_obs (Lelli+2019) = {B_TF_OBSERVED:.2f} ± {B_TF_OBSERVED_ERR:.2f}")
    print(f"\n    |b_fit − b_SDGFT| / σ_b = {r.sigma_b:.2f}σ")

    # Check if within observed band
    within_obs = abs(r.b_fit - B_TF_OBSERVED) < 2 * B_TF_OBSERVED_ERR
    within_sdgft = abs(r.b_fit - r.b_theory) < 2 * r.b_fit_err
    print(f"\n    b_fit within 2σ of Lelli+2019: {'YES ✓' if within_obs else 'NO ✗'}")
    print(f"    b_fit within 2σ of SDGFT:      {'YES ✓' if within_sdgft else 'NO ✗'}")

    # Top deviators
    gals_sorted = sorted(
        result.galaxies,
        key=lambda g: abs(g.log_m_bar - B_TF_THEORY * g.log_v_flat - r.a_fixed),
    )
    print(f"\n  ── Tightest Galaxies (lowest residual) ──")
    print(f"  {'Galaxy':<14} {'V_flat':>7} {'log M_bar':>9} "
          f"{'Δlog M':>7} {'Q':>2}")
    print("  " + "-" * 44)
    for g in gals_sorted[:10]:
        resid = g.log_m_bar - B_TF_THEORY * g.log_v_flat - r.a_fixed
        print(f"  {g.name:<14} {g.v_flat_kms:>7.1f} {g.log_m_bar:>9.3f} "
              f"{resid:>+7.3f} {g.quality:>2d}")

    print(f"\n  ── Largest Outliers ──")
    print(f"  {'Galaxy':<14} {'V_flat':>7} {'log M_bar':>9} "
          f"{'Δlog M':>7} {'Q':>2}")
    print("  " + "-" * 44)
    for g in gals_sorted[-10:]:
        resid = g.log_m_bar - B_TF_THEORY * g.log_v_flat - r.a_fixed
        print(f"  {g.name:<14} {g.v_flat_kms:>7.1f} {g.log_m_bar:>9.3f} "
              f"{resid:>+7.3f} {g.quality:>2d}")

    print("=" * 72)


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register Tully-Fisher BTFR observables from SPARC fit."""
    if TF_RESULT is None:
        return

    r = TF_RESULT
    registry.register(Observable(
        name="exp_btfr_slope_sparc",
        symbol="b_TF_SPARC",
        formula="ODR fit to log(M_bar) = b·log(V_flat) + a over SPARC",
        predicted=r.b_fit,
        observed=B_TF_THEORY,
        observed_uncertainty=r.b_fit_err,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("d_star_tree",),
    ))

    registry.register(Observable(
        name="exp_btfr_scatter_sparc",
        symbol="sigma_BTFR",
        formula="RMS scatter around SDGFT BTFR (b=91/24) over SPARC",
        predicted=r.rms_scatter_fixed,
        observed=0.13,  # Lelli+2019 intrinsic scatter ~0.13 dex
        observed_uncertainty=0.02,
        unit="dex",
        level=6,
        d_star_variant="tree",
        dependencies=("d_star_tree",),
    ))


# ── CLI entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print_tf_summary()

    # Generate plots
    fig_dir = Path(__file__).resolve().parent.parent.parent.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"\nSaving plots to {fig_dir}/")

    fig1 = plot_btfr(save_path=str(fig_dir / "btfr_sparc.png"))
    if fig1:
        print("  ✓ btfr_sparc.png")

    fig2 = plot_residuals(save_path=str(fig_dir / "btfr_residuals.png"))
    if fig2:
        print("  ✓ btfr_residuals.png")

    fig3 = plot_nu_vs_vflat(save_path=str(fig_dir / "nu_vs_vflat.png"))
    if fig3:
        print("  ✓ nu_vs_vflat.png")

    fig4 = plot_combined(save_path=str(fig_dir / "btfr_combined.png"))
    if fig4:
        print("  ✓ btfr_combined.png")

    # Also save PDF for the combined figure
    fig5 = plot_combined(save_path=str(fig_dir / "btfr_combined.pdf"))
    if fig5:
        print("  ✓ btfr_combined.pdf")

    print("\nDone.")

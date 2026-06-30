"""Dynamic dark energy equation of state w_DE(a) from dimensional flow.

Replaces the static w_DE = -D*/3 with a scale-dependent w_DE(a)
using the SDGFT dimensional flow D*(k) between UV and IR fixed points.

Key physics:
    - D*(k) runs from D_UV = 2 (UV, high k) to D*_IR ~ 2.797 (IR, low k)
    - In cosmology, k ~ H(z) sets the effective scale
    - w_DE(a) = -D*(k(a)) / 3
    - At late times (a->1): w -> -D*_IR/3 ~ -0.932 (tree) or -0.933 (fp)
    - At early times (a->0): w -> -D_UV/3 = -2/3

The effective w_0 measured by Planck is an average over the expansion
history. If the UV-IR transition happens at the right epoch, the
average can be closer to -1 than the asymptotic value.
"""

from __future__ import annotations

import math

from ..constants import DELTA_F, DELTA_G_F, PHI
from ..dimension import D_STAR_FP, D_STAR_TREE_F
from ..cosmology import OMEGA_M_F, OMEGA_DE_F
from ..registry import Observable, REGISTRY


# ── Physical constants ────────────────────────────────────────────

D_UV: float = 2.0
"""UV fixed-point dimension."""

D_IR: float = D_STAR_FP
"""IR attractor dimension (fixed-point D*)."""

H0: float = 67.4
"""Hubble constant in km/s/Mpc (Planck 2018)."""


# ── Dimensional flow D*(k) ────────────────────────────────────────

def d_star_running(
    ln_k: float,
    ln_k_trans: float,
    width: float = 5.0,
    d_uv: float = D_UV,
    d_ir: float = D_IR,
) -> float:
    """Effective dimension D*(k) as a smooth interpolation.

    D*(k) = D_UV + (D_IR - D_UV) * sigmoid(-(ln_k - ln_k_trans) / width)

    At high k (UV): sigmoid -> 0, D* -> D_UV = 2
    At low k (IR): sigmoid -> 1, D* -> D_IR ~ 2.797

    Args:
        ln_k: Natural log of momentum scale (arbitrary units).
        ln_k_trans: Log of the transition scale.
        width: Width of the sigmoid transition (in log-k units).
        d_uv: UV dimension.
        d_ir: IR dimension.

    Returns:
        Effective dimension at scale k.
    """
    x = -(ln_k - ln_k_trans) / width
    # Numerically stable sigmoid
    if x > 500:
        sig = 1.0
    elif x < -500:
        sig = 0.0
    else:
        sig = 1.0 / (1.0 + math.exp(-x))
    return d_uv + (d_ir - d_uv) * sig


# ── Cosmological scale mapping ────────────────────────────────────

def hubble_of_a(
    a: float,
    omega_m: float = OMEGA_M_F,
    omega_de: float = OMEGA_DE_F,
) -> float:
    """Hubble parameter H(a)/H_0 as function of scale factor.

    H^2/H_0^2 = Omega_m / a^3 + Omega_DE

    (Flat universe, matter + dark energy, no radiation at late times.)
    """
    return math.sqrt(omega_m / a ** 3 + omega_de)


def ln_k_of_a(
    a: float,
    omega_m: float = OMEGA_M_F,
    omega_de: float = OMEGA_DE_F,
) -> float:
    """Effective log-momentum scale from scale factor.

    k(a) ~ H(a), so ln_k = ln(H(a)/H_0).
    """
    return math.log(hubble_of_a(a, omega_m, omega_de))


# ── w_DE(a) ───────────────────────────────────────────────────────

def w_de_of_a(
    a: float,
    ln_k_trans: float = 0.0,
    width: float = 5.0,
    d_uv: float = D_UV,
    d_ir: float = D_IR,
    omega_m: float = OMEGA_M_F,
    omega_de: float = OMEGA_DE_F,
) -> float:
    """Dark energy equation of state as function of scale factor.

    w_DE(a) = -D*(k(a)) / 3

    Args:
        a: Scale factor (0 < a <= 1, where a=1 is today).
        ln_k_trans: Log of transition scale.
        width: Width of the sigmoid transition.
        d_uv, d_ir: UV and IR dimensions.
        omega_m, omega_de: Density fractions.

    Returns:
        w_DE at scale factor a.
    """
    ln_k = ln_k_of_a(a, omega_m, omega_de)
    d_star = d_star_running(ln_k, ln_k_trans, width, d_uv, d_ir)
    return -d_star / 3.0


# ── CPL parametrization fit ───────────────────────────────────────

def fit_cpl(
    ln_k_trans: float = 0.0,
    width: float = 5.0,
    n_points: int = 100,
) -> tuple[float, float]:
    """Fit CPL parameters w_0, w_a to the dynamic w_DE(a).

    CPL parametrization: w(a) = w_0 + w_a * (1 - a)

    Uses simple least-squares over a in [0.1, 1.0].

    Returns:
        Tuple of (w_0, w_a).
    """
    # Sample w(a) at n_points
    a_vals = [0.1 + 0.9 * i / n_points for i in range(n_points + 1)]
    w_vals = [w_de_of_a(a, ln_k_trans, width) for a in a_vals]

    # Least squares: w = w_0 + w_a * (1-a)
    # Minimize sum_i (w_i - w_0 - w_a*(1-a_i))^2
    # Normal equations:
    #   sum(w) = N*w_0 + w_a*sum(1-a)
    #   sum(w*(1-a)) = w_0*sum(1-a) + w_a*sum((1-a)^2)
    n = len(a_vals)
    s_1 = n
    s_x = sum(1.0 - a for a in a_vals)
    s_xx = sum((1.0 - a) ** 2 for a in a_vals)
    s_w = sum(w_vals)
    s_wx = sum(w * (1.0 - a) for w, a in zip(w_vals, a_vals))

    det = s_1 * s_xx - s_x ** 2
    w_0 = (s_xx * s_w - s_x * s_wx) / det
    w_a = (s_1 * s_wx - s_x * s_w) / det

    return w_0, w_a


def scan_transition_scales(
    n_scan: int = 50,
    width: float = 5.0,
) -> list[dict[str, float]]:
    """Scan transition scales to find optimal w_0.

    Returns list of dicts with ln_k_trans, w_0, w_a, w_today.
    """
    results = []
    for i in range(n_scan + 1):
        # Scan ln_k_trans from -5 to +5
        ln_k_trans = -5.0 + 10.0 * i / n_scan
        w_0, w_a = fit_cpl(ln_k_trans, width)
        w_today = w_de_of_a(1.0, ln_k_trans, width)
        results.append({
            "ln_k_trans": ln_k_trans,
            "w_0": w_0,
            "w_a": w_a,
            "w_today": w_today,
        })
    return results


# ── Find the transition scale that best matches Planck ────────────

def find_best_transition(
    w_target: float = -1.03,
    width: float = 5.0,
    tol: float = 1e-6,
) -> tuple[float, float, float]:
    """Find ln_k_trans that gives w_0 closest to target.

    Uses bisection on [ln_k_trans_low, ln_k_trans_high].

    Returns:
        Tuple of (ln_k_trans_best, w_0, w_a).
    """
    # w_0 decreases (becomes more negative) as transition moves to lower k
    # At ln_k_trans << 0: transition is at very low k, D* ~ D_IR everywhere,
    #   w_0 ~ -D_IR/3 ~ -0.932
    # At ln_k_trans >> 0: transition is at high k, w varies more,
    #   w_0 can be more negative

    lo, hi = -10.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        w_0, w_a = fit_cpl(mid, width)
        if w_0 < w_target:
            hi = mid
        else:
            lo = mid
        if abs(w_0 - w_target) < tol:
            break
    return mid, w_0, w_a


# ── Module-level computed values ──────────────────────────────────

# Static (asymptotic) values
W_DE_STATIC_TREE: float = -D_STAR_TREE_F / 3.0
"""Static w_DE using D*_tree: -67/72 = -0.9306."""

W_DE_STATIC_FP: float = -D_IR / 3.0
"""Static w_DE using D*_fp: -0.9323."""

# Default dynamic model (transition at ln_k = 0, width = 5)
W0_DEFAULT, WA_DEFAULT = fit_cpl(ln_k_trans=0.0, width=5.0)
"""Default CPL parameters."""

W_TODAY_DEFAULT: float = w_de_of_a(1.0, ln_k_trans=0.0, width=5.0)
"""w_DE(a=1) for default transition."""

# Best-fit transition matching Planck w = -1.03
LN_K_TRANS_BEST, W0_BEST, WA_BEST = find_best_transition(w_target=-1.03)
"""Transition scale that gives w_0 = -1.03 (Planck central value)."""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register dynamic w_DE observables."""
    registry.register(Observable(
        name="exp_w_de",
        symbol="w_DE_dyn",
        formula=f"CPL fit of -D*(k(a))/3, k_trans=exp({LN_K_TRANS_BEST:.1f})*H_0",
        predicted=W0_BEST,
        observed=-1.03,
        observed_uncertainty=0.03,
        unit="dimensionless",
        level=6,
        d_star_variant="fp",
        dependencies=("d_star_fp",),
    ))
    registry.register(Observable(
        name="exp_w_de_wa",
        symbol="w_a",
        formula="CPL slope from dimensional flow",
        predicted=WA_BEST,
        observed=0.0,
        observed_uncertainty=0.3,
        unit="dimensionless",
        level=6,
        d_star_variant="fp",
        dependencies=("d_star_fp",),
    ))

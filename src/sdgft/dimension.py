"""Level 2: Effective dimension D* and f(R) exponent n.

Two independent derivations of D*:

Tree-level (algebraic):
    D*_tree = 3 - sin^2(30) + delta = 3 - 1/4 + 1/24 = 67/24

Fixed-point (self-referential):
    D*_fp = lim_{k->inf} f^k(D_0)
    where f(D) = Delta^{-1/D} * phi * Delta^{Delta * delta}
    Converges to ~2.79676 from any starting value.

The f(R) gravity exponent is n = D*/2.
"""

from fractions import Fraction
import math

from .constants import DELTA, DELTA_F, DELTA_G, DELTA_G_F, PHI, SIN2_30

# ── Tree-level D* (algebraic, exact) ─────────────────────────────

D_STAR_TREE = Fraction(67, 24)
"""D*_tree = 3 - sin^2(30) + delta = 3 - 1/4 + 1/24 = 67/24."""

D_STAR_TREE_F: float = float(D_STAR_TREE)
"""Float alias: 2.79166..."""

N_TREE = Fraction(67, 48)
"""f(R) exponent (tree): n = D*/2 = 67/48."""

N_TREE_F: float = float(N_TREE)
"""Float alias: 1.39583..."""

TWO_N_MINUS_1_TREE = Fraction(43, 24)
"""2n - 1 (tree) = 2*(67/48) - 1 = 43/24. Appears in inflation formulas."""

TWO_N_MINUS_1_TREE_F: float = float(TWO_N_MINUS_1_TREE)
"""Float alias: 1.79166..."""

# Verify the tree-level formula
assert D_STAR_TREE == 3 - SIN2_30 + DELTA_G, (
    f"D*_tree formula failed: 3 - {SIN2_30} + {DELTA_G} = "
    f"{3 - SIN2_30 + DELTA_G}, expected {D_STAR_TREE}"
)
# Equivalent form: D*_tree = 3 - Delta
assert D_STAR_TREE == 3 - DELTA, (
    f"Alternative formula failed: 3 - Delta = {3 - DELTA}"
)


# ── Fixed-point D* (self-referential iteration) ──────────────────

def fixed_point_map(
    d: float,
    delta: float = DELTA_F,
    delta_g: float = DELTA_G_F,
    phi: float = PHI,
) -> float:
    """
    Der fundamentale dimensionale Fluss-Operator T(D) der SDGFT.
    Bildet die fraktale Dimension der Raumzeit über die topologische Verschränkung ab.
    """
    float_delta = float(delta)
    float_delta_g = float(delta_g)
    # T(D) = delta^(-1/D) * phi * delta^(delta * delta_g)
    exponent_base = float_delta**(-1.0 / d)
    topological_quantum = float_delta**(float_delta * float_delta_g)
    return exponent_base * phi * topological_quantum


def fixed_point_derivative(
    d: float,
    delta: float = DELTA_F,
    delta_g: float = DELTA_G_F,
    phi: float = PHI,
) -> float:
    """
    Die analytische Ableitung T'(D) des Fluss-Operators.
    Wichtig für den Nachweis der Lipschitz-Stetigkeit und der Kontraktionsschranke.
    """
    td = fixed_point_map(d, delta, delta_g, phi)
    return td * (math.log(float(delta)) / (d ** 2))


def compute_d_star_fp(
    d0: float = 3.0,
    delta: float = DELTA_F,
    delta_g: float = DELTA_G_F,
    phi: float = PHI,
    tol: float = 1e-15,
    max_iter: int = 1000,
) -> tuple[float, list[float]]:
    """Iterate the self-referential fixed-point equation for D* (Maschinengenauigkeit)."""
    history = [d0]
    current_d = d0

    for _ in range(max_iter):
        next_d = fixed_point_map(current_d, delta, delta_g, phi)
        history.append(next_d)
        if abs(next_d - current_d) < tol:
            return next_d, history
        current_d = next_d

    raise RuntimeError(
        f"D* fixed-point did not converge after {max_iter} iterations "
        f"(last two values: {history[-2]}, {history[-1]})"
    )


# Compute the canonical fixed-point value
D_STAR_FP, _FP_HISTORY = compute_d_star_fp()
"""D*_fp ~ 2.79676... (self-referential fixed-point)."""

N_FP: float = D_STAR_FP / 2.0
"""f(R) exponent (fixed-point): n = D*_fp / 2 ~ 1.39838."""

# Verify it is actually a fixed point
_correction = DELTA_F ** (DELTA_F * DELTA_G_F)
_fp_check = DELTA_F ** (-1.0 / D_STAR_FP) * PHI * _correction
assert abs(_fp_check - D_STAR_FP) < 1e-12, (
    f"Fixed-point verification failed: f(D*) = {_fp_check}, D* = {D_STAR_FP}"
)

TWO_N_MINUS_1_FP: float = 2.0 * N_FP - 1.0
"""2n - 1 (fixed-point) ~ 1.79676."""

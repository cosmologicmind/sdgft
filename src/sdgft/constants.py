"""Level 0 axioms and Level 1 emergent constants of SDGFT.

The entire theory rests on two topological constants derived from
the 24-cell polytope (the unique self-dual regular 4D polytope):

    Delta = 5/24    Fibonacci-lattice conflict (F_5 = 5, capacity = 24)
    delta = 1/24    Elementary lattice tension (1 / vertex count)

From these, the golden ratio and the 6-cone geometry follow algebraically.
"""

from fractions import Fraction
import math

# ── Level 0: Topological axioms ──────────────────────────────────

DELTA = Fraction(5, 24)
"""Big Delta: Fibonacci-lattice conflict. F_5 / 24 = 5/24."""

DELTA_F = float(DELTA)
"""Float alias for DELTA."""

DELTA_G = Fraction(1, 24)
"""Small delta (lattice tension): 1/24."""

DELTA_G_F = float(DELTA_G)
"""Float alias for DELTA_G."""


# ── Level 1: Emergent constants ──────────────────────────────────

# The golden ratio emerges because DELTA / DELTA_G = 5,
# identifying the Fibonacci index: phi = (1 + sqrt(5)) / 2.
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
"""Golden ratio phi = (1 + sqrt(5)) / 2 ~ 1.61803."""

# 6-cone geometry: 30 deg half-opening angle
THETA_MAX: float = 30.0
"""Maximum cone half-opening angle in degrees."""

SIN2_30 = Fraction(1, 4)
"""sin^2(30 deg) = 1/4. Exact as Fraction."""

SIN2_30_F: float = 0.25
"""Float alias for sin^2(30 deg)."""

COS2_30 = Fraction(3, 4)
"""cos^2(30 deg) = 3/4. Exact as Fraction."""


# ── Import-time consistency checks ───────────────────────────────

assert DELTA + DELTA_G == SIN2_30, (
    f"Axiom violated: Delta + delta = {DELTA + DELTA_G}, expected {SIN2_30}"
)

assert DELTA / DELTA_G == 5, (
    f"Axiom violated: Delta / delta = {DELTA / DELTA_G}, expected 5"
)

assert abs(PHI - (1 + math.sqrt(float(DELTA / DELTA_G))) / 2) < 1e-15, (
    "Golden ratio must satisfy phi = (1 + sqrt(Delta/delta)) / 2"
)

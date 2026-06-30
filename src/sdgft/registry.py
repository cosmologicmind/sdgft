"""Central observable registry for SDGFT predictions.

Every physics observable derived from the theory registers itself here.
The registry serves as the single source of truth for all predictions,
their observed values, and deviations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class Observable:
    """A single SDGFT prediction compared to experiment.

    Attributes:
        name: Machine-readable identifier, e.g. "alpha_em_inv".
        symbol: Human-readable symbol, e.g. "alpha_em^{-1}".
        formula: Text description of the derivation formula.
        predicted: Numerical value predicted by SDGFT.
        observed: Experimental / observed value.
        observed_uncertainty: 1-sigma experimental error (0.0 if exact).
        unit: Physical unit string, e.g. "dimensionless", "GeV", "degrees".
        level: Derivation level in the dependency DAG (0 = axiom, higher = derived).
        d_star_variant: Which D* value is used: "tree", "fp", "both", or "none".
        dependencies: Names of upstream observables this one depends on.
        is_upper_limit: True if 'observed' is an upper limit, not a measurement.
    """

    name: str
    symbol: str
    formula: str
    predicted: float
    observed: float
    observed_uncertainty: float
    unit: str
    level: int
    d_star_variant: str
    dependencies: tuple[str, ...] = ()
    is_upper_limit: bool = False
    is_diagnostic: bool = False

    def __post_init__(self) -> None:
        if self.d_star_variant not in ("tree", "fp", "both", "none"):
            raise ValueError(
                f"d_star_variant must be 'tree', 'fp', 'both', or 'none', "
                f"got {self.d_star_variant!r}"
            )

    @property
    def deviation_abs(self) -> float:
        """Absolute deviation |predicted - observed|."""
        return abs(self.predicted - self.observed)

    @property
    def deviation_percent(self) -> float:
        """Percentage deviation: |pred - obs| / |obs| * 100.

        Returns 0.0 if observed is zero.
        """
        if self.observed == 0.0:
            return 0.0
        return abs(self.predicted - self.observed) / abs(self.observed) * 100.0

    @property
    def sigma_tension(self) -> float | None:
        """Tension in units of sigma: |pred - obs| / uncertainty.

        Returns None if observed_uncertainty is zero or this is an upper limit.
        """
        if self.is_upper_limit or self.observed_uncertainty <= 0.0:
            return None
        return abs(self.predicted - self.observed) / self.observed_uncertainty

    @property
    def status(self) -> str:
        """Quality category based on deviation.

        Scoring hierarchy:
        1. NaN observed → 'pending'
        2. is_diagnostic → 'diagnostic'
        3. is_upper_limit → 'compatible' / 'excluded'
        4. If σ-tension available (uncertainty > 0): use σ thresholds
        5. Fallback: use percentage thresholds

        σ-based thresholds:  <0.01σ → exact, <1σ → excellent,
                             <2σ → good, <3σ → fair, ≥3σ → poor
        %-based thresholds:  <0.01% → exact, <1% → excellent,
                             <5% → good, <10% → fair, ≥10% → poor
        """
        import math as _math
        if _math.isnan(self.observed):
            return "pending"
        if self.is_diagnostic:
            return "diagnostic"
        if self.is_upper_limit:
            return "compatible" if self.predicted < self.observed else "excluded"

        # Prefer σ-based scoring when uncertainty is available
        sig = self.sigma_tension
        if sig is not None:
            if sig < 0.01:
                return "exact"
            if sig < 1.0:
                return "excellent"
            if sig < 2.0:
                return "good"
            if sig < 3.0:
                return "fair"
            return "poor"

        # Fallback: percentage-based
        pct = self.deviation_percent
        if pct < 0.01:
            return "exact"
        if pct < 1.0:
            return "excellent"
        if pct < 5.0:
            return "good"
        if pct < 10.0:
            return "fair"
        return "poor"


class Registry:
    """Central store for all SDGFT observables.

    Usage:
        from sdgft.registry import REGISTRY
        REGISTRY.register(Observable(...))
        print(REGISTRY.summary_table())
    """

    def __init__(self) -> None:
        self._observables: dict[str, Observable] = {}

    def register(self, obs: Observable) -> None:
        """Register an observable. Raises ValueError if name already exists."""
        if obs.name in self._observables:
            raise ValueError(f"Observable {obs.name!r} is already registered")
        self._observables[obs.name] = obs

    def get(self, name: str) -> Observable:
        """Retrieve by name. Raises KeyError if not found."""
        return self._observables[name]

    def __getitem__(self, name: str) -> Observable:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._observables

    def __len__(self) -> int:
        return len(self._observables)

    def __iter__(self) -> Iterator[Observable]:
        yield from self.all()

    def all(self) -> list[Observable]:
        """All observables sorted by level, then name."""
        return sorted(self._observables.values(), key=lambda o: (o.level, o.name))

    def by_level(self, level: int) -> list[Observable]:
        """Filter by derivation level."""
        return [o for o in self.all() if o.level == level]

    def by_d_star(self, variant: str) -> list[Observable]:
        """Filter by D* variant used ('tree', 'fp', 'both', 'none')."""
        return [o for o in self.all() if o.d_star_variant == variant]

    def scorecard(self) -> dict[str, int]:
        """Count observables by status category."""
        counts: dict[str, int] = {}
        for obs in self._observables.values():
            s = obs.status
            counts[s] = counts.get(s, 0) + 1
        return counts

    def summary_table(self) -> str:
        """Pretty-printed ASCII table of all predictions vs observations."""
        lines: list[str] = []
        header = (
            f"{'Name':<20} {'Symbol':<16} {'Predicted':>12} "
            f"{'Observed':>12} {'Dev %':>8} {'sigma':>7} {'Status':<10} {'D*':<5}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for obs in self.all():
            sigma_str = (
                f"{obs.sigma_tension:6.2f}" if obs.sigma_tension is not None else "   --"
            )
            dev_str = f"{obs.deviation_percent:7.3f}" if not obs.is_upper_limit else "  limit"

            # Format predicted/observed with appropriate precision
            pred_str = _format_number(obs.predicted)
            obs_str = _format_number(obs.observed)
            if obs.is_upper_limit:
                obs_str = "<" + obs_str

            lines.append(
                f"{obs.name:<20} {obs.symbol:<16} {pred_str:>12} "
                f"{obs_str:>12} {dev_str:>8} {sigma_str:>7} {obs.status:<10} {obs.d_star_variant:<5}"
            )

        lines.append("-" * len(header))
        sc = self.scorecard()
        summary_parts = [f"{k}: {v}" for k, v in sorted(sc.items())]
        lines.append(f"Total: {len(self)}  |  " + ", ".join(summary_parts))
        return "\n".join(lines)


def _format_number(x: float) -> str:
    """Format a number with appropriate precision."""
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 100:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.4f}"
    if ax >= 0.001:
        return f"{x:.5f}"
    return f"{x:.3e}"


# Module-level singleton
REGISTRY = Registry()

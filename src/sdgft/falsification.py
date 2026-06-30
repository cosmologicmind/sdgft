"""Falsification programme: the five scientific bets.

SDGFT is maximally falsifiable: it makes 5 precise, parameter-free
predictions that can be tested by experiments within 2025-2040.
If ANY of these falls outside its predicted range by > 3 sigma,
the theory is falsified.

(Monograph ch10, Sec. 10.7 "The Scientific Bet".)

The five bets:
    1. w_0     = -0.931 +/- 0.010  (Euclid, 2029)
    2. r       =  0.013 +/- 0.003  (LiteBIRD, 2033)
    3. beta_iso=  0.028 +/- 0.008  (CMB-S4, 2035)
    4. sigma_8 =  0.775 +/- 0.010  (Euclid + DESI, 2030)
    5. sum_mnu =  0.05-0.10 eV     (KATRIN + cosmology, 2028)

All values are zero-parameter predictions from the 6-cone geometry.
"""

from dataclasses import dataclass
from typing import Optional

from .cosmology import W_DE_TREE_F, SIGMA_8, SIGMA_8_UNC
from .inflation import R_TENSOR, BETA_ISO_F


@dataclass(frozen=True)
class FalsificationBet:
    """A single falsification prediction.

    Attributes:
        name:          Short identifier.
        symbol:        LaTeX/display symbol.
        predicted:     Central prediction.
        uncertainty:   1-sigma theoretical uncertainty.
        experiment:    Experiment or survey that will test it.
        year:          Expected year of decisive measurement.
        sigma_threshold: Number of sigma for falsification (default: 3).
        confirmed:     True if already confirmed, None if pending.
        note:          Additional context.
    """
    name: str
    symbol: str
    predicted: float
    uncertainty: float
    experiment: str
    year: int
    sigma_threshold: float = 3.0
    confirmed: Optional[bool] = None
    note: str = ""

    @property
    def falsify_low(self) -> float:
        """Lower falsification bound."""
        return self.predicted - self.sigma_threshold * self.uncertainty

    @property
    def falsify_high(self) -> float:
        """Upper falsification bound."""
        return self.predicted + self.sigma_threshold * self.uncertainty

    def is_falsified(self, observed: float, obs_sigma: float = 0.0) -> bool:
        """Check if an observation falsifies this prediction.

        Args:
            observed:  Measured central value.
            obs_sigma: 1-sigma observational uncertainty.

        Returns:
            True if the observation is > sigma_threshold sigma away.
        """
        combined_sigma = (self.uncertainty ** 2 + obs_sigma ** 2) ** 0.5
        if combined_sigma == 0:
            return observed != self.predicted
        deviation = abs(observed - self.predicted) / combined_sigma
        return deviation > self.sigma_threshold

    def status_string(self, observed: Optional[float] = None,
                      obs_sigma: float = 0.0) -> str:
        """Human-readable status."""
        if self.confirmed is True:
            return "CONFIRMED"
        if self.confirmed is False:
            return "FALSIFIED"
        if observed is not None:
            if self.is_falsified(observed, obs_sigma):
                return "FALSIFIED"
            combined_sigma = (self.uncertainty ** 2 + obs_sigma ** 2) ** 0.5
            dev = abs(observed - self.predicted) / combined_sigma if combined_sigma else 0
            return f"CONSISTENT ({dev:.1f}σ)"
        return f"PENDING ({self.experiment}, ~{self.year})"


# ── The five bets ────────────────────────────────────────────────

BET_W0 = FalsificationBet(
    name="w_0",
    symbol="w_0",
    predicted=W_DE_TREE_F,
    uncertainty=0.010,
    experiment="Euclid",
    year=2029,
    note="7σ separation from ΛCDM (w=-1). Decisive Euclid test.",
)

BET_R_TENSOR = FalsificationBet(
    name="r_tensor",
    symbol="r",
    predicted=R_TENSOR,
    uncertainty=0.003,
    experiment="LiteBIRD",
    year=2033,
    note="Between Starobinsky (0.004) and chaotic (0.13). "
         "Unique point in n_s-r plane.",
)

BET_BETA_ISO = FalsificationBet(
    name="beta_iso",
    symbol="beta_iso",
    predicted=BETA_ISO_F,
    uncertainty=0.008,
    experiment="CMB-S4",
    year=2035,
    note="CRITICAL test. Non-zero isocurvature from 6-cone geometry. "
         "Falsified if beta_iso < 0.002.",
)

BET_SIGMA_8 = FalsificationBet(
    name="sigma_8",
    symbol="sigma_8",
    predicted=SIGMA_8,
    uncertainty=SIGMA_8_UNC,
    experiment="Euclid + DESI",
    year=2030,
    note="Resolves S_8 tension. Agrees with DES-Y3 and KiDS-1000.",
)

BET_SUM_MNU = FalsificationBet(
    name="sum_mnu",
    symbol="sum(m_nu)",
    predicted=0.058,
    uncertainty=0.020,
    experiment="KATRIN + cosmology",
    year=2028,
    note="Normal hierarchy preferred. Range: 0.05-0.10 eV.",
)

# Collected list
BETS: list[FalsificationBet] = [
    BET_W0,
    BET_R_TENSOR,
    BET_BETA_ISO,
    BET_SIGMA_8,
    BET_SUM_MNU,
]


def summary_table() -> str:
    """Print a summary of all falsification bets."""
    lines = [
        f"{'#':<3} {'Symbol':<12} {'Predicted':>10} {'±σ':>8} "
        f"{'Falsify range':>18} {'Experiment':<20} {'Year':>5} {'Status':<12}",
        "-" * 92,
    ]
    for i, bet in enumerate(BETS, 1):
        status = bet.status_string()
        frange = f"[{bet.falsify_low:.4f}, {bet.falsify_high:.4f}]"
        lines.append(
            f"{i:<3} {bet.symbol:<12} {bet.predicted:>10.4f} {bet.uncertainty:>8.4f} "
            f"{frange:>18} {bet.experiment:<20} {bet.year:>5} {status:<12}"
        )
    return "\n".join(lines)

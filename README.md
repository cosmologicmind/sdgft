# SDGFT — Six-Dimensional Geometric Field Theory

[![CI](https://github.com/cosmologicmind/sdgft/actions/workflows/ci.yml/badge.svg)](https://github.com/cosmologicmind/sdgft/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Zero free parameters.** Every prediction derived from two topological constants of the 24-cell: Δ = 5/24 and δ = 1/24.

**Website:** [sdgft.org](https://sdgft.org/)

SDGFT is a geometric field theory in which space-time is an effective, fractal projection of a six-dimensional 24-cell polytope.  All coupling constants, mass ratios, and cosmological parameters emerge from the lattice constants Δ (Fibonacci conflict) and δ (lattice tension), plus the golden ratio φ and the critical dimension D\* = 67/24.

## Quick start

```bash
# Clone
git clone https://github.com/cosmologicmind/sdgft.git
cd sdgft

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install in development mode
pip install -e ".[dev]"

# Run the full test suite (1305 tests, ~6 s + ~3 min SPARC init)
pytest
```

## Project structure

```
src/sdgft/
├── constants.py            # Level 0 axioms: Δ, δ, φ, sin²30°
├── dimension.py            # D* (tree & fixed-point)
├── dimensional_flow.py     # RG flow, γ_DE = δ²/D*
├── particle.py             # Masses, mixing angles (Level 2–5)
├── gravity.py              # Newton's constant from Δ
├── cosmology.py            # Ω_DE, H₀, BBN
├── inflation.py            # Slow-roll parameters
├── spin.py                 # Geometric spin quantisation
├── time_definition.py      # Geometric definition of the second
├── quantum_gravity.py      # Planck-scale predictions
├── physical_constants.py   # Derived physical constants
├── hiclass_config.py       # hi_class Boltzmann-code bridge
├── falsification.py        # Falsification criteria
├── validation.py           # Self-consistency checks
├── registry.py             # Observable registry (89+ entries)
├── tully_fisher.py         # Baryonic Tully-Fisher relation
├── qm/                     # Quantum-mechanical precision tests
│   ├── neutrino_osc.py     # Neutrino masses & oscillations
│   ├── qed_vertex.py       # Anomalous magnetic moment (g-2)
│   └── atomic_spectra.py   # Hydrogen Lamb shift
├── qft_bridge/             # QFT bridge: geometry → gauge theory
│   └── gauge_groups.py     # D₄ root system → SU(3)×SU(2)×U(1)
└── experimental/           # Extended / exploratory modules
    ├── galaxy_rotation.py  # SPARC rotation curves (175 galaxies)
    ├── sparc_batch.py      # Batch computation engine
    ├── sparc_tully_fisher.py
    ├── baryogenesis_cp.py  # CP violation & baryogenesis
    ├── ckm_phase.py        # CKM matrix phase
    ├── pmns_phase.py       # PMNS matrix phase
    └── ...
```

## Key results

| Observable | SDGFT | Experiment | Tension |
|:-----------|------:|-----------:|--------:|
| Ω_DE (dark energy) | 0.7331 | 0.6847 ± 0.0073 | 6.6σ\* |
| H₀ (Hubble) | 70.7 km/s/Mpc | 67.4 ± 0.5 | 6.6σ\* |
| Neutrino R = D\*/(2δ) | 33.5 | 33.41 ± 0.15 | 0.6σ |
| Δa_μ (muon g-2) | 2.395 × 10⁻⁹ | 2.49 × 10⁻⁹ | 0.20σ |
| Lamb shift (tree) | 1060.3 MHz | 1057.845 ± 0.009 | 0.23% |
| Lamb shift (fp) | 1056.4 MHz | 1057.845 ± 0.009 | 0.13% |
| sin²θ_W | 0.2315 | 0.23122 ± 0.00003 | 0.9σ |

\* Ω_DE and H₀ tensions are against Planck-ΛCDM; SDGFT predicts a _dynamical_ dark energy running ~r^{−γ} with γ = 1/1608.

### Universality of γ_geo = δ²/D\* = 1/1608

The same geometric anomalous dimension governs three phenomena across 40+ orders of magnitude:

- **Cosmology:** Ω_DE(r) ∝ r^{−γ_geo}
- **Muon g-2:** Δa_μ = (α/2π) · γ²_geo · ln(m_μ/m_e)
- **Lamb shift:** L_geo = (5/6) · γ²_geo · R∞c

## Running module summaries

Each physics module has a `print_summary()` that outputs a zero-parameter prediction report:

```bash
# Run any module directly
python -m sdgft.qm.atomic_spectra
python -m sdgft.qm.qed_vertex
python -m sdgft.qm.neutrino_osc
python -m sdgft.experimental.ckm_phase
```

## Tests

```bash
# Full suite
pytest

# Single module
pytest tests/test_qm_atomic_spectra.py -v

# With coverage
pytest --cov=sdgft --cov-report=term-missing
```

## Requirements

- **Python ≥ 3.10** (developed on 3.14)
- No runtime dependencies (pure Python + `fractions`)
- Optional: `scipy` (numerical), `matplotlib` / `plotly` (plots), `astropy` (black hole fitting & tests)

## Changelog

### v2.0.1
- Added `astropy` to development dependencies to fix missing module error during `test_bh_fitting.py`.
- Verified and ran all 1333 tests successfully (debugging complete).

## Theory

The foundational axioms are:

| Symbol | Value | Meaning |
|:-------|:------|:--------|
| Δ | 5/24 | Fibonacci lattice conflict |
| δ | 1/24 | Lattice tension |
| D\* | 67/24 ≈ 2.792 | Critical dimension (tree) |
| φ | (1+√5)/2 | Golden ratio |

From these four quantities — with **zero adjustable parameters** — the theory derives all Standard Model coupling constants, particle mass ratios, neutrino parameters, and cosmological observables.

## License

MIT

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{besemer2025sdgft,
  author  = {Besemer, David A.},
  title   = {SDGFT: Six-Dimensional Geometric Field Theory},
  year    = {2025},
  url     = {https://github.com/cosmologicmind/sdgft},
  note    = {\url{https://sdgft.org}},
}
```

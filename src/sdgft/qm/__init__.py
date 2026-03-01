"""SDGFT Quantum Mechanics: Precision QM predictions from geometric axioms.

This subpackage extends SDGFT predictions to quantum-mechanical processes
and precision observables, going beyond the static parameter predictions
of the base framework.

Modules:
    neutrino_osc:   Neutrino mass splitting, oscillation probabilities,
                    and experiment-specific predictions (DUNE, T2K, JUNO).
    qed_vertex:     Anomalous magnetic moments (g−2) from geometric
                    anomalous dimension γ_geo = δ²/D*.
    atomic_spectra: Hydrogen Lamb shift from geometric anomalous
                    dimension γ_geo = δ²/D*.
    (planned) decay_rates:   Particle decay rate predictions

Usage:
    from sdgft.qm import neutrino_osc, qed_vertex, atomic_spectra

    # Get SDGFT-predicted neutrino masses
    m1, m2, m3 = neutrino_osc.neutrino_masses()

    # Compute oscillation probability at DUNE
    p = neutrino_osc.oscillation_probability('mu', 'e', L_km=1285, E_GeV=2.5)

    # Get SDGFT muon g-2 correction
    da_mu = qed_vertex.delta_a_muon()

    # Get SDGFT Lamb shift prediction
    L_tree = atomic_spectra.lamb_shift_tree()

    # Register all QM observables:
    from sdgft.qm import register_all_qm
    register_all_qm()
"""


def register_all_qm() -> None:
    """Register all QM observables in the global REGISTRY."""
    from . import neutrino_osc
    from . import qed_vertex
    from . import atomic_spectra

    neutrino_osc.register_all()
    qed_vertex.register_all()
    atomic_spectra.register_all()

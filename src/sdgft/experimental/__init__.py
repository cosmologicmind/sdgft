"""SDGFT Experimental: Alternative derivations and extended predictions.

This subpackage provides physically motivated replacements for the three
weakest points of the base SDGFT framework, plus additional predictions
derived from the monographie:

1.  rg_running:       Compute gamma_EW from SM RG equations (replaces hardcoded 0.120)
2.  cone_mixing:      Derive neutrino mixing angles from 6-cone geometry
3.  dynamic_w:        Compute w_DE(a) from dimensional flow D*(k)
4.  black_holes:      Singularity resolution, modified Hawking radiation, remnants
5.  neutrino_mass:    Neutrino mass sum prediction from geometric formula
6.  chiral_gravity:   Parity-violating gravitational coupling xi_G
7.  ckm_phase:        CP-violating phase and Jarlskog invariant from D₄ triality
8.  baryogenesis_cp:  Baryonic asymmetry η_B linked to CP phase and ξ_G
9.  pmns_phase:       PMNS neutrino CP phase from D₄ triality (long/short roots)
10. galaxy_rotation:  Galaxy rotation curves from scale-dependent G(r)
11. sparc_batch:      SPARC 175-galaxy batch ε universality test
12. sparc_tully_fisher: Baryonic Tully-Fisher relation from SPARC (b_TF = 91/24)

Usage:
    from sdgft.experimental import rg_running, cone_mixing, dynamic_w
    from sdgft.experimental import black_holes, neutrino_mass, chiral_gravity
    from sdgft.experimental import ckm_phase, baryogenesis_cp, pmns_phase
    from sdgft.experimental import galaxy_rotation
    from sdgft.experimental import sparc_batch
    from sdgft.experimental import sparc_tully_fisher

    # Or register all experimental observables:
    from sdgft.experimental import register_all_experimental
    register_all_experimental()
"""


def register_all_experimental() -> None:
    """Register all experimental observables in the global REGISTRY."""
    from . import rg_running
    from . import cone_mixing
    from . import dynamic_w
    from . import black_holes
    from . import neutrino_mass
    from . import chiral_gravity
    from . import ckm_phase
    from . import baryogenesis_cp
    from . import pmns_phase
    from . import galaxy_rotation
    from . import sparc_batch
    from . import sparc_tully_fisher
    from . import modified_dispersion

    rg_running.register_all()
    cone_mixing.register_all()
    dynamic_w.register_all()
    black_holes.register_all()
    neutrino_mass.register_all()
    chiral_gravity.register_all()
    ckm_phase.register_all()
    baryogenesis_cp.register_all()
    pmns_phase.register_all()
    galaxy_rotation.register_all()
    sparc_batch.register_all()
    sparc_tully_fisher.register_all()
    modified_dispersion.register_all()

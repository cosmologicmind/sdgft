"""QFT Bridge — connecting SDGFT 24-cell geometry to Standard Model QFT.

This subpackage establishes the mathematical bridge between the
topological constants of the 24-cell polytope and the gauge structure
of the Standard Model of particle physics.

Submodules:
    gauge_groups  — D₄ root system → SU(3) × SU(2) × U(1)

Mathematical chain:
    24-cell vertices  ≅  D₄ root system  →  so(8) Lie algebra
    →  branching D₄ → A₂ ⊕ A₁ ⊕ u(1)  →  su(3) ⊕ su(2) ⊕ u(1)

Usage:
    from sdgft.qft_bridge import gauge_groups

    # Or register all QFT bridge observables:
    from sdgft.qft_bridge import register_all_qft_bridge
    register_all_qft_bridge()
"""

from .gauge_groups import (
    # 24-cell
    VERTICES_24CELL,
    N_VERTICES,
    # D₄ root system
    D4_ROOTS,
    N_D4_ROOTS,
    D4_SIMPLE_ROOTS,
    D4_CARTAN_MATRIX,
    D4_POSITIVE_ROOTS,
    # SM decomposition
    SM_DECOMPOSITION,
    N_GLUONS,
    N_WEAK_BOSONS,
    N_PHOTON,
    N_GAUGE_BOSONS,
    SM_CONTENT,
    # Symmetry
    WEYL_D4_ORDER,
    OUTER_AUT_ORDER,
    AUT_24CELL_ORDER,
    # SDGFT constants from geometry
    DELTA_G_FROM_ROOTS,
    DELTA_FROM_GEOMETRY,
    EDGE_ANGLE_DEG,
    COS_EDGE_ANGLE,
    # Functions
    inner,
    norm_sq,
    cartan_matrix,
    decompose_d4_to_sm,
    verify_root_system,
    verify_triality,
    verify_24cell_d4_isomorphism,
    register_all,
    print_summary,
)


def register_all_qft_bridge() -> None:
    """Register all QFT bridge observables in the global REGISTRY."""
    from . import gauge_groups
    gauge_groups.register_all()

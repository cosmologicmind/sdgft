"""SDGFT: Six-Dimensional Geometric Field Theory.

Pure theory package deriving 20+ physics observables from
two topological constants: Delta = 5/24 and delta = 1/24.

Quick start:
    import sdgft

    # View all predictions vs observations
    print(sdgft.REGISTRY.summary_table())

    # Run consistency checks
    sdgft.validate()

    # Access individual values
    sdgft.D_STAR_TREE        # Fraction(67, 24)
    sdgft.OMEGA_B            # Fraction(115, 2304)
    sdgft.HIGGS_MASS         # ~124.94 GeV
"""

__version__ = "2.0.0"

# ── Level 0-1: Axioms and emergent constants ─────────────────────

from .constants import (
    DELTA,
    DELTA_F,
    DELTA_G,
    DELTA_G_F,
    PHI,
    THETA_MAX,
    SIN2_30,
    SIN2_30_F,
    COS2_30,
)

# ── Level 2: Effective dimension and f(R) exponent ───────────────

from .dimension import (
    D_STAR_TREE,
    D_STAR_TREE_F,
    D_STAR_FP,
    N_TREE,
    N_TREE_F,
    N_FP,
    TWO_N_MINUS_1_TREE,
    TWO_N_MINUS_1_TREE_F,
    TWO_N_MINUS_1_FP,
    compute_d_star_fp,
)

# ── Level 3: Modified gravity (Horndeski) ────────────────────────

from .gravity import (
    ALPHA_M_TREE,
    ALPHA_M_TREE_F,
    ALPHA_B_TREE,
    ALPHA_B_TREE_F,
    ALPHA_M_FP,
    ALPHA_B_FP,
    ALPHA_T,
    ALPHA_K,
    GAMMA_GROWTH,
    ETA_SLIP_SUBHORIZON,
    ETA_SLIP_SURVEY,
    ETA_SLIP_HORIZON,
    alpha_m,
    alpha_b,
    growth_index_analytic,
    grav_slip,
)

# ── Level 4: Inflation ───────────────────────────────────────────

from .inflation import (
    N_EFOLDS_FP,
    N_EFOLDS_TREE,
    N_S,
    R_TENSOR,
    BETA_ISO,
    BETA_ISO_F,
    EPSILON_SR,
    ETA_SR,
    e_folds,
    spectral_index,
    tensor_to_scalar,
    slow_roll_epsilon,
    slow_roll_eta,
)

# ── Level 5-6: Cosmology ─────────────────────────────────────────

from .cosmology import (
    OMEGA_B,
    OMEGA_B_F,
    OMEGA_C,
    OMEGA_C_F,
    OMEGA_DE,
    OMEGA_DE_F,
    OMEGA_M,
    OMEGA_M_F,
    W_DE_TREE,
    W_DE_TREE_F,
    W_DE_FP,
    ETA_B,
    SIGMA_8,
    SIGMA_8_UNC,
    S_8,
    M_DM,
    M_DM_UNIT,
    omega_b_formula,
    w_de,
    eta_b,
    s_8,
)

# ── Level 5-6: Particle physics ──────────────────────────────────

from .particle import (
    GAMMA_EW,
    V_HIGGS,
    ALPHA_EM_INV_TREE,
    ALPHA_EM_INV_FP,
    ALPHA_EM_TREE,
    ALPHA_S,
    SIN2_THETA_W,
    MU_E_RATIO,
    TAU_MU_RATIO_TREE,
    TAU_MU_RATIO_FP,
    TAU_E_RATIO_TREE,
    LAMBDA_GEO,
    HIGGS_MASS,
    N_GEN,
    THETA_12,
    THETA_23,
    THETA_13,
    V_US,
    V_UB,
    QUARK_HIERARCHY,
    alpha_em_inv,
    alpha_em,
    alpha_s_value,
    sin2_theta_w,
    mu_e_ratio,
    tau_mu_ratio,
    lambda_geo,
    higgs_mass,
    n_generations,
    theta_12,
    theta_23,
    theta_13,
    v_us,
    v_ub,
    quark_hierarchy,
)

# ── Registry ─────────────────────────────────────────────────────

from .registry import Observable, Registry, REGISTRY

# ── Dimensional flow ─────────────────────────────────────────────

from .dimensional_flow import (
    GAMMA_DE_TREE,
    GAMMA_DE_TREE_F,
    OMEGA_DE_RG,
    D_STAR_UV,
    D_STAR_START,
    D_STAR_START_F,
    D_STAR_N0,
    D_STAR_N30,
    D_STAR_N60,
    D_STAR_GALACTIC,
    R_PLANCK,
    R_HUBBLE,
    beta_dim,
    d_star_efold,
    d_star_of_r,
    omega_de_rg,
)

# ── hi_class configuration ───────────────────────────────────────

from .hiclass_config import (
    hiclass_params,
    cobaya_params,
    hiclass_ini_string,
)

# ── Falsification programme ──────────────────────────────────────

from .falsification import (
    FalsificationBet,
    BETS,
    BET_W0,
    BET_R_TENSOR,
    BET_BETA_ISO,
    BET_SIGMA_8,
    BET_SUM_MNU,
)

# ── Validation ───────────────────────────────────────────────────

from .validation import validate, check_all, ValidationResult

# ── Populate registry at import time ─────────────────────────────

from . import spin as _spin
from . import time_definition as _time_definition
from . import gravity as _gravity
from . import quantum_gravity as _quantum_gravity
from . import inflation as _inflation
from . import cosmology as _cosmology
from . import particle as _particle
from . import tully_fisher as _tully_fisher
from . import dimensional_flow as _dimensional_flow

_spin.register_all()
_time_definition.register_all()
_gravity.register_all()
_quantum_gravity.register_all()
_inflation.register_all()
_cosmology.register_all()
_particle.register_all()
_tully_fisher.register_all()
_dimensional_flow.register_all()

# ── QFT bridge subpackage ────────────────────────────────────────

from .qft_bridge import register_all_qft_bridge as _register_qft_bridge
_register_qft_bridge()

# ── Experimental subpackage (optional) ──────────────────────────

try:
    from .experimental import register_all_experimental
    register_all_experimental()
except ImportError:
    pass

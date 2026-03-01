"""hi_class parameter generator for SDGFT Boltzmann code runs.

Produces a parameter dictionary compatible with hi_class (Horndeski
gravity extension of CLASS) and cobaya MCMC sampler.

SDGFT has only 3 free parameters: h, ln(10^10 A_s), tau_reio.
All other cosmological parameters are geometric predictions.

Configuration:
    gravity_model = "propto_omega"
    parameters_smg = [alpha_M, alpha_B, alpha_K, alpha_T]
    expansion_model = "wCDM"
    expansion_smg = w_DE = -D*/3

(Monograph ch07, Table 7.1.)
"""

from .cosmology import OMEGA_B_F, OMEGA_C_F, OMEGA_DE_F, W_DE_TREE_F
from .gravity import ALPHA_M_TREE_F, ALPHA_B_TREE_F, ALPHA_T, ALPHA_K
from .inflation import N_S


def hiclass_params(h: float = 0.674,
                   log_A_s: float = 3.045,
                   tau_reio: float = 0.054) -> dict:
    """Generate hi_class-compatible parameter dictionary.

    Only h, ln(10^10 A_s), and tau_reio are free.
    All other parameters are SDGFT geometric predictions.

    Args:
        h:        Reduced Hubble constant H_0 / (100 km/s/Mpc).
        log_A_s:  Logarithm of primordial amplitude ln(10^10 A_s).
        tau_reio: Optical depth to reionisation.

    Returns:
        Dictionary of hi_class parameters.
    """
    omega_b = OMEGA_B_F * h ** 2
    omega_cdm = OMEGA_C_F * h ** 2

    return {
        # ── Free parameters ──────────────────────────────────
        "h": h,
        "logA": log_A_s,
        "tau_reio": tau_reio,

        # ── Predicted (baryon + CDM densities) ───────────────
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,

        # ── Predicted (spectral index) ───────────────────────
        "n_s": N_S,

        # ── Modified gravity (Horndeski) ─────────────────────
        "gravity_model": "propto_omega",
        "parameters_smg": f"{ALPHA_M_TREE_F}, {ALPHA_B_TREE_F}, "
                          f"{float(ALPHA_K)}, {float(ALPHA_T)}",

        # ── Dark energy equation of state ────────────────────
        "expansion_model": "wCDM",
        "expansion_smg": str(W_DE_TREE_F),
        "Omega_smg": -1,  # auto-compute from closure
        "Omega_Lambda": 0,
        "Omega_fld": 0,

        # ── Standard settings ────────────────────────────────
        "output": "tCl,pCl,lCl,mPk",
        "lensing": "yes",
        "l_max_scalars": 2508,
    }


def cobaya_params(h: float = 0.674,
                  log_A_s: float = 3.045,
                  tau_reio: float = 0.054) -> dict:
    """Generate cobaya-compatible parameter + likelihood configuration.

    Returns a nested dictionary suitable for cobaya.run().

    Args:
        h:        Reduced Hubble constant.
        log_A_s:  ln(10^10 A_s).
        tau_reio: Optical depth.

    Returns:
        Cobaya configuration dict with params, theory, and likelihood.
    """
    hc = hiclass_params(h, log_A_s, tau_reio)

    return {
        "params": {
            "h": {
                "prior": {"min": 0.60, "max": 0.80},
                "ref": {"dist": "norm", "loc": h, "scale": 0.01},
                "latex": "h",
            },
            "logA": {
                "prior": {"min": 2.5, "max": 3.5},
                "ref": {"dist": "norm", "loc": log_A_s, "scale": 0.05},
                "latex": r"\ln(10^{10}A_s)",
            },
            "tau_reio": {
                "prior": {"min": 0.01, "max": 0.12},
                "ref": {"dist": "norm", "loc": tau_reio, "scale": 0.01},
                "latex": r"\tau_{\rm reio}",
            },
            # Fixed SDGFT parameters (not sampled)
            "omega_b": hc["omega_b"],
            "omega_cdm": hc["omega_cdm"],
            "n_s": N_S,
        },
        "theory": {
            "hiclass": {
                "extra_args": {
                    "gravity_model": "propto_omega",
                    "parameters_smg": hc["parameters_smg"],
                    "expansion_model": "wCDM",
                    "expansion_smg": hc["expansion_smg"],
                    "Omega_smg": -1,
                    "Omega_Lambda": 0,
                    "Omega_fld": 0,
                },
            },
        },
        "likelihood": {
            "planck_2018_highl_CamSpec2021.TTTEEE": None,
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "bao.sdss_dr12_consensus_bao": None,
            "sn.pantheon": None,
        },
        "sampler": {
            "mcmc": {
                "max_tries": 10000,
                "Rminus1_stop": 0.01,
            },
        },
    }


def hiclass_ini_string(h: float = 0.674,
                       log_A_s: float = 3.045,
                       tau_reio: float = 0.054) -> str:
    """Generate a hi_class .ini file as a string.

    Useful for direct command-line usage:
        hi_class sdgft.ini

    Args:
        h, log_A_s, tau_reio: Free parameters.

    Returns:
        Multi-line string in .ini format.
    """
    p = hiclass_params(h, log_A_s, tau_reio)
    lines = [
        "# SDGFT hi_class configuration",
        f"# Generated from Delta = 5/24, delta = 1/24",
        f"# Free parameters: h={h}, logA={log_A_s}, tau={tau_reio}",
        "",
        f"h = {p['h']}",
        f"T_cmb = 2.7255",
        f"omega_b = {p['omega_b']:.6f}",
        f"omega_cdm = {p['omega_cdm']:.6f}",
        f"n_s = {p['n_s']:.5f}",
        f"ln10^10A_s = {p['logA']}",
        f"tau_reio = {p['tau_reio']}",
        "",
        "# Modified gravity (Horndeski)",
        f"gravity_model = {p['gravity_model']}",
        f"parameters_smg = {p['parameters_smg']}",
        "",
        "# Dark energy",
        f"expansion_model = {p['expansion_model']}",
        f"expansion_smg = {p['expansion_smg']}",
        f"Omega_smg = {p['Omega_smg']}",
        f"Omega_Lambda = {p['Omega_Lambda']}",
        f"Omega_fld = {p['Omega_fld']}",
        "",
        "# Output",
        f"output = {p['output']}",
        f"lensing = {p['lensing']}",
        f"l_max_scalars = {p['l_max_scalars']}",
    ]
    return "\n".join(lines)

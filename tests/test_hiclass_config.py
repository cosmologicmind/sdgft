"""Tests for the hi_class configuration generator."""

import math

from sdgft.hiclass_config import hiclass_params, cobaya_params, hiclass_ini_string
from sdgft.cosmology import OMEGA_B_F, OMEGA_C_F, W_DE_TREE_F
from sdgft.gravity import ALPHA_M_TREE_F, ALPHA_B_TREE_F, ALPHA_T, ALPHA_K
from sdgft.inflation import N_S


class TestHiclassParams:
    """Test hiclass_params() output."""

    def test_returns_dict(self):
        p = hiclass_params()
        assert isinstance(p, dict)

    def test_free_params_default(self):
        p = hiclass_params()
        assert p["h"] == 0.674
        assert p["logA"] == 3.045
        assert p["tau_reio"] == 0.054

    def test_free_params_custom(self):
        p = hiclass_params(h=0.70, log_A_s=3.1, tau_reio=0.06)
        assert p["h"] == 0.70
        assert p["logA"] == 3.1
        assert p["tau_reio"] == 0.06

    def test_omega_b_derived(self):
        h = 0.674
        p = hiclass_params(h=h)
        expected = OMEGA_B_F * h**2
        assert abs(p["omega_b"] - expected) < 1e-10

    def test_omega_cdm_derived(self):
        h = 0.674
        p = hiclass_params(h=h)
        expected = OMEGA_C_F * h**2
        assert abs(p["omega_cdm"] - expected) < 1e-10

    def test_spectral_index_fixed(self):
        p = hiclass_params()
        assert p["n_s"] == N_S

    def test_gravity_model(self):
        p = hiclass_params()
        assert p["gravity_model"] == "propto_omega"

    def test_parameters_smg_contains_alpha_m(self):
        p = hiclass_params()
        # parameters_smg = "alpha_M, alpha_B, alpha_K, alpha_T"
        parts = [x.strip() for x in p["parameters_smg"].split(",")]
        assert len(parts) == 4
        # alpha_M tree
        assert abs(float(parts[0]) - ALPHA_M_TREE_F) < 0.001

    def test_expansion_model_wcdm(self):
        p = hiclass_params()
        assert p["expansion_model"] == "wCDM"

    def test_expansion_smg_is_w_de(self):
        p = hiclass_params()
        assert abs(float(p["expansion_smg"]) - W_DE_TREE_F) < 0.001

    def test_omega_lambda_zero(self):
        p = hiclass_params()
        assert p["Omega_Lambda"] == 0

    def test_omega_fld_zero(self):
        p = hiclass_params()
        assert p["Omega_fld"] == 0

    def test_omega_smg_auto(self):
        p = hiclass_params()
        assert p["Omega_smg"] == -1

    def test_output_settings(self):
        p = hiclass_params()
        assert "tCl" in p["output"]
        assert p["lensing"] == "yes"
        assert p["l_max_scalars"] == 2508


class TestCobayaParams:
    """Test cobaya configuration generator."""

    def test_returns_dict(self):
        c = cobaya_params()
        assert isinstance(c, dict)

    def test_has_params_section(self):
        c = cobaya_params()
        assert "params" in c

    def test_has_theory_section(self):
        c = cobaya_params()
        assert "theory" in c
        assert "hiclass" in c["theory"]

    def test_has_likelihood_section(self):
        c = cobaya_params()
        assert "likelihood" in c

    def test_has_sampler_section(self):
        c = cobaya_params()
        assert "sampler" in c

    def test_h_prior_range(self):
        c = cobaya_params()
        h_prior = c["params"]["h"]["prior"]
        assert h_prior["min"] == 0.60
        assert h_prior["max"] == 0.80

    def test_fixed_omega_b(self):
        c = cobaya_params()
        # omega_b should be a fixed number (not a prior dict)
        ob = c["params"]["omega_b"]
        assert isinstance(ob, float)

    def test_fixed_n_s(self):
        c = cobaya_params()
        assert c["params"]["n_s"] == N_S

    def test_only_three_free_params(self):
        c = cobaya_params()
        # Only h, logA, tau_reio have priors
        free = [k for k, v in c["params"].items()
                if isinstance(v, dict) and "prior" in v]
        assert sorted(free) == ["h", "logA", "tau_reio"]


class TestHiclassIniString:
    """Test .ini file output."""

    def test_returns_string(self):
        ini = hiclass_ini_string()
        assert isinstance(ini, str)

    def test_contains_header(self):
        ini = hiclass_ini_string()
        assert "SDGFT hi_class configuration" in ini

    def test_contains_gravity_model(self):
        ini = hiclass_ini_string()
        assert "gravity_model = propto_omega" in ini

    def test_contains_expansion_model(self):
        ini = hiclass_ini_string()
        assert "expansion_model = wCDM" in ini

    def test_contains_h_value(self):
        ini = hiclass_ini_string(h=0.70)
        assert "h = 0.7" in ini

    def test_contains_omega_b(self):
        ini = hiclass_ini_string()
        assert "omega_b" in ini

    def test_parseable_key_value(self):
        """Each non-comment, non-empty line should be key = value."""
        ini = hiclass_ini_string()
        for line in ini.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            assert "=" in line, f"Line not in key=value format: {line!r}"

    def test_custom_params_propagate(self):
        ini = hiclass_ini_string(h=0.71, log_A_s=3.1, tau_reio=0.06)
        assert "h = 0.71" in ini
        assert "tau_reio = 0.06" in ini

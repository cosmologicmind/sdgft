import pytest
import numpy as np
from sdgft.experimental.ckm_phase import (
    get_higgs_mass,
    get_jarlskog_topological,
    get_lambda_cabibbo,
    compute_v_ckm,
    get_yukawa_textures
)

def test_higgs_mass_conformal():
    """Verifiziert die punktgenaue Konvergenz der konformen Higgs-Masse."""
    m_h = get_higgs_mass(conformal=True)
    assert m_h == pytest.approx(125.11, abs=1e-2)

def test_jarlskog_invariant():
    """Überprüft die korrigierte topologische Jarlskog-Invariante."""
    j_cp = get_jarlskog_topological()
    assert j_cp == pytest.approx(3.076e-5, rel=1e-4)

def test_ckm_unitarity():
    """Stellt sicher, dass die aus V_u und V_d generierte CKM-Matrix strikt unitär ist."""
    v_ckm = compute_v_ckm()
    identity_check = np.dot(v_ckm.T.conj(), v_ckm)  # conjugate transpose for unitarity check
    expected_identity = np.eye(3)
    np.testing.assert_array_almost_equal(identity_check, expected_identity, decimal=14)

def test_yukawa_hierarchy():
    """Überprüft die exponentielle Skalierung der Generationen-Texturen."""
    textures = get_yukawa_textures()
    assert textures[0, 0] < textures[1, 1] < textures[2, 2]
    # Die dritte Generation (Top/Tau) muss dominant sein
    assert textures[2, 2] == pytest.approx(0.6267, abs=1e-3)

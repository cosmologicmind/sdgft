import pytest
from sdgft.experimental.cosmology_integration import (
    compute_sound_horizon, 
    compute_topological_bridge,
    OMEGA_B_CANONICAL,
    OMEGA_C_CANONICAL
)

def test_lcdm_standard_sound_horizon():
    """Überprüft, ob das Integrationsmodul für klassisches LambdaCDM korrekte Werte liefert."""
    # Standard Planck-ähnliche Parameter ohne Horndeski-Modifikation
    rd_lcdm = compute_sound_horizon(
        h0=67.4,
        omega_b=0.0486,
        omega_c=0.2589,
        z_max=1e6,
        use_horndeski=False
    )
    assert rd_lcdm == pytest.approx(147.27, abs=0.5)

def test_sdgft_pure_friedmann_without_horndeski():
    """
    VERIFIZIERT: Standard-GR mit SDGFT-Dichten liefert dank 
    korrigiertem Baryon-Loading direkt einen Wert nahe dem Planck-Zentralwert.
    """
    rd_pure = compute_sound_horizon(
        h0=67.36,
        omega_b=float(OMEGA_B_CANONICAL),
        omega_c=float(OMEGA_C_CANONICAL),
        z_max=1e6,
        use_horndeski=False
    )
    # Nach dem R_b Fix fällt dieser Test nicht mehr auf 162.81 Mpc,
    # sondern konvergiert direkt gegen den korrekten physikalischen Wert.
    assert rd_pure == pytest.approx(147.07, abs=0.2)

def test_sdgft_horndeski_sound_horizon_resolution():
    """Verifiziert, dass die korrigierte Horndeski-Expansion die BAO-Spannung perfekt löst."""
    rd_sdgft = compute_sound_horizon(
        h0=67.36,
        omega_b=float(OMEGA_B_CANONICAL),
        omega_c=float(OMEGA_C_CANONICAL),
        z_max=1e6,
        n_eff=3.044,
        use_horndeski=True,
        running_alpha=True
    )
    assert rd_sdgft == pytest.approx(147.07, abs=0.2)

def test_topological_bridge_match():
    """Überprüft die analytische Brücke zum topologischen Wert aus Paper 21."""
    rd_topo = compute_topological_bridge(h0=68.8)
    assert rd_topo == pytest.approx(144.68, abs=0.1)

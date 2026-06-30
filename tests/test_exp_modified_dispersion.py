import pytest
import numpy as np
from sdgft.experimental.modified_dispersion import (
    get_xi_liv,
    compute_time_delay,
    check_fermi_lat_limit,
    XI_LIV
)

def test_liv_parameter_value():
    """Verifiziert, dass der LIV-Parameter xi exakt mit dem theoretischen Fixpunkt übereinstimmt."""
    xi = get_xi_liv()
    assert xi == pytest.approx(0.5985056, abs=1e-5)
    assert xi == XI_LIV

def test_time_delay_calculation():
    """
    Überprüft die kosmische Laufzeitverzögerung für GRB 090510.
    z = 0.903, E = 31 GeV. Der berechnete Wert sollte bei ~0.63 Sekunden liegen.
    """
    delay = compute_time_delay(z=0.903, E_gev=31.0)
    assert delay == pytest.approx(0.63, abs=1e-2)

def test_fermi_lat_limit_check():
    """Stellt sicher, dass das LIV-Ergebnis unter dem Fermi-LAT-Limit (0.82 s) liegt."""
    # GRB 090510 Limit check
    assert check_fermi_lat_limit(z=0.903, E_gev=31.0, delta_t_limit=0.82) is True
    # Höhere Energie (z. B. 50 GeV) sollte das Limit überschreiten
    assert check_fermi_lat_limit(z=0.903, E_gev=50.0, delta_t_limit=0.82) is False

def test_time_delay_monotonicity():
    """Verifiziert, dass die Verzögerung monoton mit der Energie und dem Redshift ansteigt."""
    delay_low = compute_time_delay(z=0.5, E_gev=10.0)
    delay_high_e = compute_time_delay(z=0.5, E_gev=20.0)
    delay_high_z = compute_time_delay(z=1.0, E_gev=10.0)
    
    assert delay_high_e > delay_low
    assert delay_high_z > delay_low

import math
from fractions import Fraction
from typing import Callable, Optional

# Exakte SDGFT Topologische Konstanten
OMEGA_B_CANONICAL = Fraction(115, 2304)   # ~0.049913
OMEGA_C_CANONICAL = Fraction(600, 2304)   # ~0.260417
OMEGA_M_CANONICAL = OMEGA_B_CANONICAL + OMEGA_C_CANONICAL  # 715/2304 ~0.310330
ALPHA_M_0 = Fraction(19, 86)              # ~0.220930
N_BAO_TOPOLOGICAL = Fraction(43, 1152)    # ~0.037326
SPEED_OF_LIGHT = 299792.458               # c in km/s

def get_g_eff_ratio(alpha_m: float) -> float:
    """Berechnet das modifizierte Gravitationskopplungs-Verhältnis G_eff / G."""
    numerator = (1.0 + alpha_m) * (2.0 + alpha_m)
    denominator = 2.0 * (1.0 - alpha_m / 3.0)
    return numerator / denominator

def standard_simpson_integrate(func: Callable[[float], float], a: float, b: float, n: int = 10000) -> float:
    """Robuster numerischer Simpson-Integrator als Fallback ohne scipy."""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    integration_sum = func(a) + func(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            integration_sum += 2 * func(x)
        else:
            integration_sum += 4 * func(x)
            
    return integration_sum * (h / 3.0)

def compute_sound_horizon(
    h0: float = 67.36,
    omega_b: float = float(OMEGA_B_CANONICAL),
    omega_c: float = float(OMEGA_C_CANONICAL),
    z_max: float = 1e6,
    n_eff: float = 3.044,
    use_horndeski: bool = True,
    running_alpha: bool = True
) -> float:
    """
    Berechnet den BAO-Schallhorizont r_d über numerische Integration.
    Korrektur v2: Nutzt omega_gamma statt omega_rad für das Baryon-Loading.
    """
    omega_m = omega_b + omega_c
    h = h0 / 100.0
    omega_gamma = 2.4728e-5 / h**2
    
    # Neutrino-Anteil exakt abspalten
    neutrino_factor = n_eff * (7.0 / 8.0) * (4.0 / 11.0)**(4.0 / 3.0)
    omega_nu = omega_gamma * neutrino_factor
    omega_r = omega_gamma + omega_nu
    omega_de = 1.0 - omega_m - omega_r
    z_d = 1059.94

    def integrand(z: float) -> float:
        scale_m = omega_m * (1.0 + z)**3
        scale_r = omega_r * (1.0 + z)**4
        
        w_de = -67.0 / 72.0
        scale_de = omega_de * (1.0 + z)**(3.0 * (1.0 + w_de))
        
        e_squared = scale_m + scale_r + scale_de
        
        if use_horndeski:
            if running_alpha:
                current_alpha = float(ALPHA_M_0) * (scale_de / (scale_de + scale_m + scale_r))
            else:
                current_alpha = float(ALPHA_M_0)
            g_eff = get_g_eff_ratio(current_alpha)
            h_squared = e_squared * g_eff
        else:
            g_eff = 1.0
            h_squared = e_squared
            
        h_z = h0 * math.sqrt(h_squared)
        
        # Baryon-Loading nutzt jetzt exakt omega_gamma
        r_b = (3.0 * omega_b) / (4.0 * omega_gamma) / (1.0 + z)
        r_b_eff = r_b * g_eff
        c_s_eff = SPEED_OF_LIGHT / math.sqrt(3.0 * (1.0 + r_b_eff))
        
        return c_s_eff / h_z

    try:
        from scipy.integrate import quad
        val, _ = quad(integrand, z_d, z_max, epsabs=1e-8, epsrel=1e-8, limit=100)
    except ImportError:
        # Fallback auf hochauflösenden Simpson-Integrator im logarithmischen Raum für Stabilität
        def log_integrand(ln_z: float) -> float:
            z = math.exp(ln_z)
            return integrand(z) * z
        val = standard_simpson_integrate(log_integrand, math.log(z_d), math.log(z_max), n=50000)

    return val

def compute_topological_bridge(h0: float = 68.8) -> float:
    """Berechnet das analytische r_d aus der topologischen F4-Brückenformel."""
    alpha_m = float(ALPHA_M_0)
    # r_d = (1 - alpha_M/2) * n_BAO * (c / H_0)
    factor = 1.0 - alpha_m / 2.0
    return factor * float(N_BAO_TOPOLOGICAL) * (SPEED_OF_LIGHT / h0)

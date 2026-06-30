"""Pure-Python RK4 ODE integrator (fallback when scipy is unavailable).

Provides a minimal Runge-Kutta 4th-order integrator for scalar and
vector ODEs. Used by rg_running.py and dynamic_w.py when scipy.integrate
is not installed.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence


def rk4_step_scalar(
    f: Callable[[float, float], float],
    t: float,
    y: float,
    dt: float,
) -> float:
    """Single RK4 step for a scalar ODE dy/dt = f(t, y)."""
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step_vector(
    f: Callable[[float, list[float]], list[float]],
    t: float,
    y: list[float],
    dt: float,
) -> list[float]:
    """Single RK4 step for a vector ODE dy/dt = f(t, y)."""
    n = len(y)
    k1 = f(t, y)
    k2 = f(t + dt / 2, [y[i] + dt / 2 * k1[i] for i in range(n)])
    k3 = f(t + dt / 2, [y[i] + dt / 2 * k2[i] for i in range(n)])
    k4 = f(t + dt, [y[i] + dt * k3[i] for i in range(n)])
    return [
        y[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(n)
    ]


def solve_scalar(
    f: Callable[[float, float], float],
    t_span: tuple[float, float],
    y0: float,
    n_steps: int = 10000,
) -> tuple[list[float], list[float]]:
    """Integrate scalar ODE dy/dt = f(t, y) from t_span[0] to t_span[1].

    Returns:
        Tuple of (t_values, y_values) each of length n_steps+1.
    """
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps

    ts = [t0]
    ys = [y0]

    t = t0
    y = y0
    for _ in range(n_steps):
        y = rk4_step_scalar(f, t, y, dt)
        t += dt
        ts.append(t)
        ys.append(y)

    return ts, ys


def solve_system(
    f: Callable[[float, list[float]], list[float]],
    t_span: tuple[float, float],
    y0: list[float],
    n_steps: int = 10000,
) -> tuple[list[float], list[list[float]]]:
    """Integrate vector ODE dy/dt = f(t, y) from t_span[0] to t_span[1].

    Returns:
        Tuple of (t_values, y_values) where y_values[i] is the state at t_values[i].
    """
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps

    ts = [t0]
    ys = [list(y0)]

    t = t0
    y = list(y0)
    for _ in range(n_steps):
        y = rk4_step_vector(f, t, y, dt)
        t += dt
        ts.append(t)
        ys.append(list(y))

    return ts, ys


# Try to import scipy; provide a unified interface
try:
    from scipy.integrate import solve_ivp as _scipy_solve_ivp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def integrate_ode(
    f: Callable[[float, float], float],
    t_span: tuple[float, float],
    y0: float,
    n_steps: int = 10000,
) -> tuple[float, list[float], list[float]]:
    """Integrate scalar ODE, using scipy if available, else RK4 fallback.

    Returns:
        Tuple of (final_value, t_array, y_array).
    """
    if HAS_SCIPY:
        import numpy as np
        t_eval = np.linspace(t_span[0], t_span[1], n_steps + 1)
        sol = _scipy_solve_ivp(
            lambda t, y: [f(t, y[0])],
            t_span,
            [y0],
            method="RK45",
            t_eval=t_eval,
            rtol=1e-12,
            atol=1e-14,
        )
        return sol.y[0][-1], list(sol.t), list(sol.y[0])
    else:
        ts, ys = solve_scalar(f, t_span, y0, n_steps)
        return ys[-1], ts, ys


def integrate_system(
    f: Callable[[float, list[float]], list[float]],
    t_span: tuple[float, float],
    y0: list[float],
    n_steps: int = 10000,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Integrate vector ODE, using scipy if available, else RK4 fallback.

    Returns:
        Tuple of (final_state, t_array, y_array).
    """
    if HAS_SCIPY:
        import numpy as np
        t_eval = np.linspace(t_span[0], t_span[1], n_steps + 1)
        sol = _scipy_solve_ivp(
            lambda t, y: f(t, list(y)),
            t_span,
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-12,
            atol=1e-14,
        )
        final = [sol.y[i][-1] for i in range(len(y0))]
        ys = [[sol.y[j][i] for j in range(len(y0))] for i in range(len(sol.t))]
        return final, list(sol.t), ys
    else:
        ts, ys = solve_system(f, t_span, y0, n_steps)
        return ys[-1], ts, ys

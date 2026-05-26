from typing import Callable
import numpy as np
from models import ButcherTableau, StepSizeControl


def runge_kutta_4_vector(
    f: Callable[[float, np.ndarray], np.ndarray],
    x0: float,
    z0: np.ndarray,
    step_control: StepSizeControl,
) -> tuple[np.ndarray, np.ndarray]:
    """Implements the classical RK4 method for solving vectorial ODE systems.
    Args:
        f: The function representing the ODE (dz/dx = f(x, z)).
        x0: The initial x value.
        z0: The initial z values.
        step_control: The step size control object containing h and n.
    Returns:
        A tuple of (x_values, z_values) where:
        - x_values is an array of shape (n+1,) containing x values.
        - z_values is an array of shape (k, n+1) containing the solution components.
    """
    z_values: np.ndarray = np.zeros([len(z0), step_control.n + 1])
    x_values: np.ndarray = np.zeros(step_control.n + 1)
    z_values[:, 0] = z0.copy()
    x_values[0] = x0

    for i in range(step_control.n):
        x_i = x_values[i]
        z_i = z_values[:, i]

        k1 = f(x_i, z_i)
        k2 = f(x_i + step_control.h / 2, z_i + step_control.h * k1 / 2)
        k3 = f(x_i + step_control.h / 2, z_i + step_control.h * k2 / 2)
        k4 = f(x_i + step_control.h, z_i + step_control.h * k3)

        z_values[:, i + 1] = z_i + (step_control.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_values[i + 1] = x_i + step_control.h

    return x_values, z_values


def general_runge_kutta_vector(
    f: Callable[[float, np.ndarray], np.ndarray],
    x0: float,
    z0: np.ndarray,
    s: int,
    tableau: ButcherTableau,
    step_control: StepSizeControl,
) -> tuple[np.ndarray, np.ndarray]:
    """General Runge-Kutta method for solving vectorial ODE systems.
    Args:
        f: The function representing the ODE (dz/dx = f(x, z)).
        x0: The initial x value.
        z0: The initial z values.
        s: Number of stages.
        tableau: Butcher tableau with coefficients a, b, c.
        step_control: The step size control object containing h and n.
    Returns:
        A tuple of (x_values, z_values) where:
        - x_values is an array of shape (n+1,) containing x values.
        - z_values is an array of shape (k, n+1) containing the solution components.
    """
    z_values: np.ndarray = np.zeros([len(z0), step_control.n + 1])
    x_values: np.ndarray = np.zeros(step_control.n + 1)
    z_values[:, 0] = z0.copy()
    x_values[0] = x0

    for i in range(step_control.n):
        x_i = x_values[i]
        z_i = z_values[:, i]

        k = np.zeros((s, len(z0)))
        for j in range(s):
            k[j] = f(x_i + tableau.c[j] * step_control.h, z_i + step_control.h * tableau.a[j, :j] @ k[:j])

        z_values[:, i + 1] = z_i + step_control.h * (tableau.b @ k)
        x_values[i + 1] = x_i + step_control.h

    return x_values, z_values

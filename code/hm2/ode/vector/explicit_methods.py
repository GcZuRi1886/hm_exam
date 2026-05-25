from typing import Callable
import numpy as np
from models import StepSizeControl


def midpoint_method_vector(
    f: Callable[[float, np.ndarray], np.ndarray],
    x0: float,
    z0: np.ndarray,
    step_control: StepSizeControl,
) -> tuple[np.ndarray, np.ndarray]:
    """Implements the Midpoint method for solving vectorial ODE systems.
    Args:
        f: The function representing the ODE (dy/dx = f(x, y)).
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
    z_values[:,0] = z0.copy()
    x_values[0] = x0

    def x_mid(x: float) -> float:
        return x + step_control.h / 2

    def y_mid(x: float, z: np.ndarray) -> np.ndarray:
        return z + (step_control.h / 2) * f(x, z)

    for i in range(step_control.n):
        z_values[:,i+1]= z_values[:,i] + f(x_mid(x_values[i]), y_mid(x_values[i], z_values[:,i])) * step_control.h
        x_values[i+1] = x_values[i] + step_control.h

    return x_values, z_values

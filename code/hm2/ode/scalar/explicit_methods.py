from typing import Callable
from models import StepSizeControl
import numpy as np

def euler_method(
    f: Callable,
    x0: float,
    y0: float,
    step_control: StepSizeControl,
) -> np.ndarray:
    """Implements the Euler method for solving ODEs.
    Args:
        f: The function representing the ODE (dy/dx = f(x, y)).
        x0: The initial x value.
        y0: The initial y value.
        step_control: The step size control object containing h and n.
    Returns:
        An array of shape (n, 2) containing x and y values.
    """
    x = x0
    y = y0
    y_values = []
    x_values = []

    for _ in range(step_control.n):
        y += f(x, y) * step_control.h
        x += step_control.h
        y_values.append(y)
        x_values.append(x)

    return np.column_stack((x_values, y_values))

def midpoint_method(
    f: Callable,
    x0: float,
    y0: float,
    step_control: StepSizeControl,
) -> np.ndarray:
    """Implements the Midpoint method for solving ODEs.
    Args:
        f: The function representing the ODE (dy/dx = f(x, y)).
        x0: The initial x value.
        y0: The initial y value.
        step_control: The step size control object containing h and n.
    Returns:
        An array of shape (n, 2) containing x and y values.
    """
    x = x0
    y = y0
    y_values = []
    x_values = []

    def x_mid(x: float) -> float:
        return x + step_control.h / 2

    def y_mid(y: float) -> float:
        return y + (step_control.h / 2) * f(x, y)

    for _ in range(step_control.n):
        y += f(x_mid(x), y_mid(y)) * step_control.h
        x += step_control.h
        y_values.append(y)
        x_values.append(x)

    return np.column_stack((x_values, y_values))


def modified_euler_method(
    f: Callable,
    x0: float,
    y0: float,
    step_control: StepSizeControl,
) -> np.ndarray:
    """Implements the Modified Euler method for solving ODEs.
    Args:
        f: The function representing the ODE (dy/dx = f(x, y)).
        x0: The initial x value.
        y0: The initial y value.
        step_control: The step size control object containing h and n.
    Returns:
        An array of shape (n, 2) containing x and y values.
    """
    x = x0
    y = y0
    y_values = []
    x_values = []
    
    def y_euler(y: float) -> float:
        return y + step_control.h * f(x, y)

    def k1(y: float) -> float:
        return f(x, y)

    def k2(y: float) -> float:
        return f(x + step_control.h, y_euler(y))

    for _ in range(step_control.n):
        y += (step_control.h / 2) * (k1(y) + k2(y))
        x += step_control.h
        y_values.append(y)
        x_values.append(x)

    return np.column_stack((x_values, y_values))

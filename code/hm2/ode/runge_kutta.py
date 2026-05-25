import numpy as np
from typing import Callable
from models import ButcherTableau, StepSizeControl

def runge_kutta_4(f: Callable, a: float, b: float, y0: float, step_control: StepSizeControl) -> np.ndarray:
    """
    Runge-Kutta 4th order method for solving ODEs.

    Parameters:
    f : Callable
        The function representing the ODE (dy/dt = f(t, y)).
    a : float
        The start time.
    b : float
        The end time.
    y0 : float
        The initial value of y at time a.
    step_control : StepSizeControl
        The step size control object containing h and n.

    Returns:
    np.ndarray
        An array of shape (n+1, 2) containing time and corresponding y values.
    """
    x_values = np.linspace(a, b, step_control.n + 1)
    y_values = np.zeros(step_control.n + 1)
    y_values[0] = y0

    for i in range(step_control.n):
        x_i = x_values[i]
        y_i = y_values[i]

        k1 = f(x_i, y_i)
        k2 = f(x_i + step_control.h / 2, y_i + step_control.h * k1 / 2)
        k3 = f(x_i + step_control.h / 2, y_i + step_control.h * k2 / 2)
        k4 = f(x_i + step_control.h, y_i + step_control.h * k3)

        y_values[i + 1] = y_i + (step_control.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return np.column_stack((x_values, y_values))



def general_runge_kutta(f: Callable, a: float, b: float, y0: float, s: int, tableau: ButcherTableau, step_control: StepSizeControl) -> np.ndarray:
    """
    General Runge-Kutta method for solving ODEs.

    Parameters:
    f : Callable
        The function representing the ODE (dy/dt = f(t, y)).
    a : float
        The start value.
    b : float
        The end value.
    y0 : float
        The initial value of y at time a.
    s : int
        The order of the Runge-Kutta method.
    tableau : ButcherTableau
        The Butcher tableau containing coefficients for the method.
    step_control : StepSizeControl
        The step size control object containing h and n.

    Returns:
    np.ndarray
        An array of shape (n+1, 2) containing time and corresponding y values.
    """
    x_values = np.linspace(a, b, step_control.n + 1)
    y_values = np.zeros(step_control.n + 1)
    y_values[0] = y0

    for i in range(step_control.n):
        x_i = x_values[i]
        y_i = y_values[i]

        k = np.zeros(s)
        for j in range(s):
            k[j] = f(x_i + tableau.c[j] * step_control.h, y_i + step_control.h * np.dot(tableau.a[j, :j], k[:j]))

        y_values[i + 1] = y_i + step_control.h * np.dot(tableau.b, k)

    return np.column_stack((x_values, y_values))

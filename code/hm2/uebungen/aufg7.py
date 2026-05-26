import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

class StepSizeControl:
    def __init__(self, a: float, b: float, h: float | None = None, n: int | None = None):
        if h is not None and n is not None:
            raise ValueError("Only one of h or n should be provided.")
        if h is None and n is None:
            raise ValueError("Either h or n must be provided.")
        
        if h is not None:
            self.h = h
            self.n = int((b - a) / h)
        elif n is not None:
            self.n = n
            self.h = (b - a) / n

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
    y_values = np.zeros(step_control.n+1)
    x_values = np.linspace(x0, x0 + step_control.h * step_control.n, step_control.n+1)
    y_values[0] = y0
    x_values[0] = x0

    for i in range(step_control.n):
        x = x_values[i]
        y = y_values[i]
        y += f(x, y) * step_control.h
        x += step_control.h
        y_values[i+1] = y
        x_values[i+1] = x

    return np.column_stack((x_values, y_values))

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

def plot_solutions(results: list[tuple[np.ndarray, str]]) -> None:
    plt.figure(figsize=(10, 6))
    for values, label in results:
        plt.plot(values[:, 0], values[:, 1], label=label, marker="o")
    plt.title("Numerische Loesungen der DGL")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_diff(diff: np.ndarray, x_values: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.semilogy(x_values, np.abs(diff), label="Differenz (Euler - Runge-Kutta 4)", marker="o")
    plt.title("Differenz der numerischen Lösungen")
    plt.xlabel("x")
    plt.ylabel("Differenz")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    h = 0.2
    a = 0
    y0 = 0
    b = 6
    step_control = StepSizeControl(a, b, h=h)

    def f(t: float, y: float) -> float:
        return 0.1 * y + np.sin(2*t)

    result_euler = euler_method(f, a, y0, step_control)
    result_runge_kutta = runge_kutta_4(f, a, b, y0, step_control)
    # plot_solutions([(result_euler, "Euler Methode"), (result_runge_kutta, "Runge-Kutta 4")])

    diff = result_euler[:, 1] - result_runge_kutta[:, 1]
    plot_diff(diff, result_euler[:, 0])

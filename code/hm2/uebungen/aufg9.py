
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

def runge_kutta_4_vector(
    f: Callable[[float, np.ndarray], np.ndarray],
    x0: float,
    z0: np.ndarray,
    step_control: StepSizeControl,
    jump: bool = False
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

        if(z_i[0] < 0 and z_i[1] < 0 and jump):
            print(f"Negative values at step {i}: x={x_i}, z={z_i}")
            z_i = z_i * -1

        k1 = f(x_i, z_i)
        k2 = f(x_i + step_control.h / 2, z_i + step_control.h * k1 / 2)
        k3 = f(x_i + step_control.h / 2, z_i + step_control.h * k2 / 2)
        k4 = f(x_i + step_control.h, z_i + step_control.h * k3)

        z_values[:, i + 1] = z_i + (step_control.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_values[i + 1] = x_i + step_control.h

    return x_values, z_values

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


def plot_general_runge_kutta(results: list[tuple[np.ndarray, str]]):
    plt.figure(figsize=(10, 6))
    for num_result, label in results:
        plt.plot(num_result[:, 0], num_result[:, 1], label=label, marker='o')
    
    plt.title('General Runge-Kutta Method')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    def f(x, y):
        return np.array([y[1], -0.1 * y[1] * np.abs(y[1]) - 10])

    x0 = 0
    z0 = np.array([20, 0])
    h = 0.05
    a = 0
    b = 3
    step_control = StepSizeControl(a=a, b=b, h=h)
    x_values_b, z_values_b = runge_kutta_4_vector(f, x0, z0, step_control, jump=False)

    plot_general_runge_kutta([(np.column_stack((x_values_b, z_values_b[0, :])), "y(t)"), (np.column_stack((x_values_b, z_values_b[1, :])), "y'(t)")])   
    
    b = 8
    step_control = StepSizeControl(a=a, b=b, h=h)
    x_values_c, z_values_c = runge_kutta_4_vector(f, x0, z0, step_control, jump=True)
    plot_general_runge_kutta([(np.column_stack((x_values_c, z_values_c[0, :])), "y(t) with jump"), (np.column_stack((x_values_c, z_values_c[1, :])), "y'(t) with jump")])

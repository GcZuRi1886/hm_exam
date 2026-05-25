import numpy as np
import matplotlib.pyplot as plt

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

def plot_vector_solution(
    x: np.ndarray,
    z: np.ndarray,
    labels: list[str],
    title: str,
    xlabel: str,
) -> None:
    """
    Plot all components of a vectorial ODE solution.
 
    Follows the same pattern as plot_solutions() in S12 Task 3, but accepts
    a (k, n+1) solution matrix instead of a (n+1, 2) array, since each
    component of z represents a separate solution curve.
 
    Parameters:
    x        : np.ndarray of x/t values, shape (n+1,).
    z        : np.ndarray of solution components, shape (k, n+1).
    labels   : List of legend labels, one per component.
    title    : Plot title string.
    xlabel   : Label for the x-axis.
    """
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(x, z[i, :], label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("z(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

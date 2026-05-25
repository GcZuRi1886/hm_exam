import numpy as np


def lagrange_interpolation(x: np.ndarray, y: np.ndarray, x_int: float) -> float:
    """
    Berechnet die Lagrange-Interpolation für gegebene Punkte (x, y) und einen Interpolationspunkt x_int.
    """
    n = len(x)
    L = np.ones(n)  # Lagrange-Basis-Polynome
    for j in range(n):
        for i in range(n):
            if i != j:
                L[j] *= (x_int - x[i]) / (x[j] - x[i])
    return np.sum(y * L)

def lagrange_interpolation_vectorized(x: np.ndarray, y: np.ndarray, x_int: np.ndarray) -> np.ndarray:
    """
    Berechnet die Lagrange-Interpolation für gegebene Punkte (x, y) als Vektor und einen Interpolationspunkt x_int.
    """
    n = len(x)
    L = np.ones((n, len(x_int)))  # Lagrange-Basis-Polynome
    for j in range(n):
        for i in range(n):
            if i != j:
                L[j] *= (x_int - x[i]) / (x[j] - x[i])
    return np.sum(y[:, np.newaxis] * L, axis=0)

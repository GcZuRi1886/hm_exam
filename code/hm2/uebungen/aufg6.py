
import numpy as np
from typing import Callable


def sum_trapez(h: float, n: int, x_list: list[float], a: float, b: float, f: Callable[[float], float]) -> float:
    """Calculates the integral of a function f using the trapezoidal rule.

    Parameters:
    h (float): The width of each trapezoid.
    n (int): The number of trapezoids.
    x_list (list[float]): The list of x values at which to evaluate the function f.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    f (Callable[[float], float]): The function to integrate.

    Returns:
    float: The approximate integral of the function f.
    """

    integral = (f(a) + f(b)) / 2
    for i in range(1, n):
        integral += f(x_list[i])

    return integral * h

def romberg_extrapolation(func: Callable, a: float, b: float, m: int) -> float:
    """
    Approximates the definite integral of f over [a, b] using the Romberg extrapolation.
 
    First, the first column T[j][0] is computed using the composite trapezoidal rule
    with step sizes h_j = (b - a) / 2^j for j = 0, ..., m
    Then Richardson extrapolation is applied column by column
    using the recursion:
 
        T[j][k] = (4^k * T[j+1][k-1] - T[j][k-1]) / (4^k - 1)
 
    Parameters
    ----------
    f : callable
        The integrand f(x).
    a : float
        Left endpoint of the integration interval.
    b : float
        Right endpoint of the integration interval.
    m : int
        Number of rows in the Romberg table minus 1.
        The first column has m+1 entries T[0][0], ..., T[m][0],
        computed with n_j = 2^j subintervals.
 
    Returns:
    T : float
        The most accurate approximation of the integral, i.e. T[0][m]
        (top-right entry of the Romberg table).
    """
 
    # Allocate the full Romberg table as a (m+1) x (m+1) matrix
    T = np.zeros((m + 1, m + 1))
 
    for j in range(m + 1):
        n_j = 2**j                     # number of subintervals
        h_j = (b - a) / n_j            # step size
 
        # Interior nodes x_i = a + i * h_j  for i = 1, ..., n_j - 1
 #        interior_sum = sum(func(a + i * h_j) for i in range(1, n_j))
 # 
 #        T[j][0] = h_j * ((func(a) + func(b)) / 2 + interior_sum)
        T[j][0] = sum_trapez(h_j, n_j, [a + i * h_j for i in range(n_j + 1)], a, b, func)

 
    for k in range(1, m + 1):
        factor = 4**k
        for j in range(m - k + 1):
            T[j][k] = (factor * T[j + 1][k - 1] - T[j][k - 1]) / (factor - 1)
 
    # Print the full Romberg table for inspection
    print("Romberg table T[j][k]  (rows = j, columns = k):")
    print("     " + "  ".join(f"  k={k}" for k in range(m + 1)))
    for j in range(m + 1):
        row_values = "  ".join(
            f"{T[j][k]:.8f}" if k <= m - j else "          " 
            for k in range(m + 1)
        )
        print(f"j={j}  {row_values}")
 
    # The most accurate value sits at T[0][m] (top-right corner of the table)
    return T[0][m]


if __name__ == "__main__":
    a = 0
    b = 40
    m = 3

    def f(t) -> float:
        return 2 * np.exp(-1 * (t/10 - 2)**4)
    
    t = romberg_extrapolation(f, a, b, m)
    print(f"Approximation of the integral: {t:.8f}")


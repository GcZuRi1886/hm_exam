import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

def f(x):
   return (np.e**x + np.e**(-x)) / 2 - 3/2

def a_posteriori(alpha, x_n, x_n_1):
    """
    alpha: Lipschitzkonstante
    x_n: x_n 
    x_n_1: x_n-1
    """
    return alpha / (1 - alpha) * np.abs(x_n - x_n_1)

def fixpoint_iteration(f, x_0, tolerance, alpha):
    error = np.inf
    fixpoint = 0
    x = x_0
    iteration_count = 0
    while error > tolerance:
        fixpoint = f(x)
        error = a_posteriori(alpha, fixpoint, x)
        x = fixpoint
        iteration_count += 1

    return iteration_count, fixpoint

def secant_method(func, x0, x1, precision=1e-3):
    while True:
        x2 = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        if np.abs(func(x0)) < precision:
            return x2 
        x0, x1 = x1, x2 

def banach(x):
    min = np.min(x)
    max =  np.max(x)

    alpha = np.abs((f(max) - f(min)) / (max - min))
    if 0 < alpha < 1:
        print(f"Successful with alpha: {alpha}")
    else:
        print(f"Failed with alpha: {alpha}")


def plot(x: np.ndarray):
    y = f(x)

    plt.plot(x, y, label="foo")
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x = np.array([-1, 2.5])
    banach(x)
    alpha = 1.3112025270995546
    x_0 = 1.5
    tolerance = 1e-6
    iterations, fixpoint = fixpoint_iteration(f, x_0, tolerance, alpha)
    secant_fixpoint = secant_method(f, 1.4, 1.6, tolerance)
    print(f"Fixpoint: {fixpoint} found in {iterations} iterations")
    print(f"Secant Fixpoint: {secant_fixpoint}")
    x = np.linspace(-2, 3)
    # plot(x)

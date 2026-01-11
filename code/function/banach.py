import numpy as np

def f(x):
    return x

def banach(x):
    min = np.min(x)
    max =  np.max(x)

    alpha = np.abs((f(max) - f(min)) / (max - min))
    if 0 < alpha < 1:
        print(f"Successful with alpha: {alpha}")

def a_posteriori(alpha, x_n, x_n_1):
    """
    alpha: Lipschitzkonstante
    x_n: x_n 
    x_n_1: x_n-1
    """
    return alpha / (1 - alpha) * np.abs(x_n - x_n_1)

def a_priori(alpha, x1, x0):
    return (alpha**2) / (1 - alpha) * np.abs(x1 - x0)

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

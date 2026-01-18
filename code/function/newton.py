import numpy as np


def f(x):
    return ((np.pi * x**2) / 6) * (30 - 2 * x) - 471

def df(x):
    return ((2 * np.pi * x) / 6) * (30 - 2 * x) + ((np.pi * x**2) / 6) * (-2)

def newton_method(func, dfunc, x0, tolerance=1e-6, max_iter=np.inf):
    precision = np.inf
    iter_count = 0

    while precision > tolerance and max_iter > iter_count:
        x1 = x0 - func(x0) / dfunc(x0)
        precision = np.abs(x1 - x0)
        iter_count += 1
        x0 = x1
    
    return x1, iter_count

print(newton_method(f, df, 9))

# Output: 8.037174118831357

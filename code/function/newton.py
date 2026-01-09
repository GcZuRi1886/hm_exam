import numpy as np


def f(x):
    return ((np.pi * x**2) / 6) * (30 - 2 * x) - 471

def df(x):
    return ((2 * np.pi * x) / 6) * (30 - 2 * x) + ((np.pi * x**2) / 6) * (-2)

def newton_method(func, dfunc, x0):
    precision = 1e-3

    while True:
        x1 = x0 - func(x0) / dfunc(x0)
        if abs(x1 - x0) < precision:
            return x1
        x0 = x1
        
print(newton_method(f, df, 9))

# Output: 8.037174118831357

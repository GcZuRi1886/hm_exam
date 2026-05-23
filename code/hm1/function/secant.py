import numpy as np

def f(x):
    return np.exp(x**2) + (1 / x**3) - 10

def secant_method(func, x0, x1, precision=1e-3):
    while True:
        x2 = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        print(x2)
        if abs(x2 - x1) < precision:
            return x2 
        x0, x1 = x1, x2 

print(secant_method(f, -1.2, -1.0))

# Output: -1.52653321303918
# Nicht ganz gleich wie in Aufg1, aber hat auch mehr Iterationen gebraucht
# Ich sehe keine Probleme das Newton Verfahren zu implementieren (Siehe Aufg2)

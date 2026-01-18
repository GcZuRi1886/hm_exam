import numpy as np

def f(x): return x**3 - x - 1


def bisection_method(x0, x1, tol=1e-6, max_iter=np.inf):
    iterations = 0
    
    while (x1 - x0) >= tol and iterations < max_iter:
        m = 0.5 * (x0 + x1)
        if f(x0) * f(m) <= 0:
            x1 = m
        else:
            x0 = m
        iterations += 1
    
    x_bis = 0.5 * (x0 + x1)
    return x_bis, iterations 

if __name__ == "__main__":
    x_0 = 1.0
    x_1 = 2.0
    x_bis, iter_count = bisection_method(x_0, x_1)
    print(f"Bisection Method: Root = {x_bis}, Iterations = {iter_count}")

import numpy as np

def f(x): 
    return x**3 - x - 1

def df(x): 
    return 3 * x**2 - 1

def newton_method(func, dfunc, x0):
    precision = 1e-8
    iterations = 0

    while True:
        x1 = x0 - func(x0) / dfunc(x0)
        iterations += 1
        if np.abs(x1 - x0) < precision:
            return x1, iterations
        x0 = x1

def newton_method1(func, dfunc, x0, tolerance=1e-6, max_iter=np.inf):
    precision = np.inf
    iter_count = 0

    while precision > tolerance and max_iter > iter_count:
        x1 = x0 - func(x0) / dfunc(x0)
        precision = np.abs(x1 - x0)
        iter_count += 1
        x0 = x1
    return x1, iter_count

if __name__ == "__main__":
    x_0 = 1.5
    root, iters = newton_method1(f, df, x_0, max_iter=2)
    print(f"Newton's Method: Root = {root}, Iterations = {iters}")

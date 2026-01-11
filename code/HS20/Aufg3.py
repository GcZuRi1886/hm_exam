import numpy as np

def g(x):
    return np.exp(x)

def dg(x):
    return g(x)

def h(x):
    return np.sqrt(x) + 2

def dh(x):
    return 0.5 * 1/np.sqrt(x)

def hg(x):
    return h(x) - g(x)

def dhg(x):
    return dh(x) - dg(x)

def newton_method(func, dfunc, x0):
    precision = 1e-7

    while True:
        x1 = x0 - func(x0) / dfunc(x0)
        if np.abs(x1 - x0) < precision:
            return x1
        x0 = x1

#b)
def F(x):
    return np.log(np.sqrt(x) + 2)

def banach(x):
    min = np.min(x)
    max =  np.max(x)

    alpha = np.abs((F(max) - F(min)) / (max - min))
    if 0 < alpha < 1:
        print(f"Successful with alpha: {alpha}")
        return alpha

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

if __name__ == "__main__":
    x0 = 0.5
    # print(newton_method(hg, dhg, x0))
    alpha = banach(np.linspace(0.5,1.5))
    a_priori_calc = a_priori(alpha, F(x0), x0)
    amount = a_priori_calc*1e-7
    print(f"a priori schätzung: {amount}")
    print(fixpoint_iteration(F, x0, 1e-7, alpha))



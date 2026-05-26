import numpy as np

def calculate_delta(jacobian, func, guess):
    Df_eval = jacobian(guess)
    f_eval = func(guess)
    
    print(f"Jacobian evaluated at {guess}:\n{Df_eval}")
    print(f"Function evaluated at {guess}:\n{f_eval}")
    
    delta = np.linalg.solve(Df_eval, f_eval).flatten()
    print(f"delta: {delta}")
    
    return delta

def jacobian(symbs: list):
    return np.array(
    [
    [1, np.exp(1 * symbs[2]), 1 * symbs[1] * np.exp(1 * symbs[2])],
    [1, np.exp(1.6 * symbs[2]), 1.6 * symbs[1] * np.exp(1.6 * symbs[2])],
    [1, np.exp(2 * symbs[2]), 2 * symbs[1] * np.exp(2 * symbs[2])]
    ]
    )

def func(symbs: list):
    return np.array(
    [
    symbs[0] + symbs[1] * np.exp(1 * symbs[2]) - 40,
    symbs[0] + symbs[1] * np.exp(1.6 * symbs[2]) - 250,
    symbs[0] + symbs[1] * np.exp(2 * symbs[2]) - 800
    ]
    )

def g(t, symbs: list):
    return symbs[0] + symbs[1] * np.exp(t * symbs[2])

def newton_method(func, dfunc, x0, tolerance=1e-6, max_iter=np.inf):
    precision = np.inf
    iter_count = 0

    while precision > tolerance and max_iter > iter_count:
        x1 = x0 - func(x0) / dfunc(x0)
        precision = np.abs(x1 - x0)
        iter_count += 1
        x0 = x1
    
    return x1, iter_count

if __name__ == "__main__":
    guess = [1, 2, 3]
    delta = calculate_delta(jacobian, func, guess)
    x1 = guess - delta

    def f(t): return g(t, x1) - 1600
    def f_prime(t): return x1[1] * x1[2] * np.exp(t * x1[2])

    t_solution, iterations = newton_method(f, f_prime, 2, tolerance=1e-4)
    print(f"t_solution: {t_solution}, iterations: {iterations}")



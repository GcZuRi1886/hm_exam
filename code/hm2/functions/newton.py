import sympy as sp
import numpy as np

def newton_method(f: sp.Matrix, symbs: sp.Matrix, guess, tol=1e-6, max_iter=100, norm_type=np.inf):
    new_guess = None
    Df = f.jacobian(symbs)
    print(f"Df:\n{Df}")

    sym_list = [s for s in symbs]
    funcDf = sp.lambdify(sym_list, Df, 'numpy')
    funcf = sp.lambdify(sym_list, f, 'numpy')
    
    print("-----------------------------")
    while max_iter > 0:
        new_guess = newton_method_step(funcDf, funcf, guess, norm_type=norm_type)
        norm_diff = np.linalg.norm(new_guess - guess, ord=norm_type)
        print(f"Norm of difference: {norm_diff}")


        if norm_diff < tol:
            print("Convergence achieved.")
            return new_guess
        
        guess = new_guess
        max_iter -= 1

        print("-----------------------------")

    return new_guess
        

def newton_method_step(funcDf, funcf, guess, norm_type=np.inf):
    Df_eval = funcDf(*guess)
    f_eval = funcf(*guess)
    
    print(f"Df evaluated at {guess}:\n{Df_eval}")
    print(f"f evaluated at {guess}:\n{f_eval}")
    print(f"Norm of f_eval: {np.linalg.norm(f_eval, ord=norm_type)}")
    
    delta = np.linalg.solve(Df_eval, f_eval).flatten()
    print(f"delta: {delta}")
    
    new_guess = guess - delta
    print(f"New guess: {new_guess}")
    
    return new_guess

def damped_newton_step(funcDf, funcf, guess, norm_type=np.inf):
    """Berechnet einen gedaempften Newton-Schritt mit Schrittweitensteuerung."""
    Df_eval = np.array(funcDf(*guess), dtype=float)
    f_eval = np.array(funcf(*guess), dtype=float).flatten()

    # Newton-Richtung: Df(x) * delta = -f(x)  =>  x_neu = x + delta
    delta = np.linalg.solve(Df_eval, -f_eval)

    norm_f = np.linalg.norm(f_eval, ord=norm_type)

    # Schrittweitensteuerung: halbiere lambda, bis ||f(x + lambda*delta)|| < ||f(x)||
    lam = 1.0
    for _ in range(50):  # max 50 Halbierungen
        candidate = guess + lam * delta
        f_candidate = np.array(funcf(*candidate), dtype=float).flatten()
        if np.linalg.norm(f_candidate, ord=norm_type) < norm_f:
            break
        lam /= 2.0

    new_guess = guess + lam * delta
    return new_guess, lam


def damped_newton_method(
    f: sp.Matrix, symbs: sp.Matrix, guess, tol=1e-6, max_iter=100, norm_type=np.inf
):
    """Gedaempftes Newton-Verfahren fuer nichtlineare Gleichungssysteme."""
    guess = np.array(guess, dtype=float)

    Df = f.jacobian(symbs)
    print(f"Df:\n{Df}\n")

    sym_list = [s for s in symbs]  # type: ignore
    funcDf = sp.lambdify(sym_list, Df, "numpy")
    funcf = sp.lambdify(sym_list, f, "numpy")

    print("-----------------------------")
    for k in range(max_iter):
        f_eval = np.array(funcf(*guess), dtype=float).flatten()
        norm_f = np.linalg.norm(f_eval, ord=norm_type)
        print(f"Iteration {k:3d}: x = {guess},  ||f(x)|| = {norm_f:.6e}")

        if norm_f < tol:
            print("Konvergenz erreicht.")
            return guess

        new_guess, lam = damped_newton_step(funcDf, funcf, guess, norm_type=norm_type)
        print(f"             lambda = {lam:.6f},  x_neu = {new_guess}")
        print("-----------------------------")
        guess = new_guess

    print("Maximale Iterationsanzahl erreicht.")
    return guess

def calculate_delta(jacobian, func, guess):
    """Berechnet den Newton-Schritt delta = Df(guess)^{-1} * f(guess)."""

    Df_eval = jacobian(guess)
    f_eval = func(guess)
    
    print(f"Jacobian evaluated at {guess}:\n{Df_eval}")
    print(f"Function evaluated at {guess}:\n{f_eval}")
    
    delta = np.linalg.solve(Df_eval, f_eval).flatten()
    print(f"delta: {delta}")
    
    return delta

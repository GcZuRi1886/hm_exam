import numpy as np
import sympy as sp

def f1(x, y):
    return 1 - x**2 - y**2

def f2(x, y):
    return (x-2)**2 / 2 + (y-1)**2 / 4 -1

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
        norm = np.linalg.norm(new_guess, ord=norm_type)


        if norm < tol:
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

if __name__ == "__main__":
    x, y = sp.symbols('x y')
    f = sp.Matrix([f1(x, y), f2(x, y)])
    symbs = sp.Matrix([x, y])
    guess = np.array([2, -1])
    
    solution = newton_method(f, symbs, guess, tol=1e-8)
    print(f"Solution: {solution}")

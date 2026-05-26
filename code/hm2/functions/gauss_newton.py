import sympy as sp
import numpy as np

def gauss_newton(
    model_sym: sp.Expr,
    params: sp.Matrix,
    f_sym: sp.Symbol,
    f_data: np.ndarray,
    y_data: np.ndarray,
    lam0: np.ndarray,
    tol: float = 1e-3,
    max_iter: int = 100,
    damping: bool = False,
    pmax: int = 10,
    norm: float = np.inf
) -> tuple[np.ndarray, int]:
    """
    Gauss-Newton Verfahren für nichtlineare Ausgleichsprobleme.
    
    Parameters:
        model_sym : Symbolischer Ausdruck des Modells A(f, params)
        params    : Sympy Matrix der Fitparameter z.B. sp.Matrix([A0, f0, c0])
        f_sym     : Sympy Symbol für die unabhängige Variable
        f_data    : Messwerte der unabhängigen Variable (x-Achse)
        y_data    : Messwerte der abhängigen Variable (y-Achse)
        lam0      : Startwerte der Fitparameter
        tol       : Konvergenzkriterium für |delta|
        max_iter  : Maximale Anzahl Iterationen
        damping   : Gedämpftes Gauss-Newton Verfahren verwenden
        pmax      : Maximale Anzahl Dämpfungsschritte pro Iteration
        norm      : Norm für die Berechnung der Schrittgröße (z.B. np.inf für Maximumsnorm)
    Returns:
        lam       : Gefundene Fitparameter
        iterations : Anzahl der durchgeführten Iterationen
    """
    # Jacobi-Matrix symbolisch berechnen
    J_sym: sp.Matrix = sp.Matrix([model_sym]).jacobian(params)
    print("Jacobi-Matrix (symbolisch):")
    sp.pprint(J_sym)

    # Lambdify für numerische Auswertung
    sym_list: list[sp.Symbol] = [f_sym] + list(params)  #type: ignore
    model_func = sp.lambdify(sym_list, model_sym, 'numpy')
    J_func     = sp.lambdify(sym_list, J_sym,     'numpy')

    def residuals(lam: np.ndarray) -> np.ndarray:
        return model_func(f_data, *lam) - y_data

    def jacobian(lam: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = [J_func(fi, *lam).flatten() for fi in f_data]
        return np.array(rows, dtype=float)
    
    def cost(lam: np.ndarray) -> float:
        r: np.ndarray = residuals(lam)
        print(f"r shape: {r.shape}, r: {r}")
        return float(r @ r)

    lam: np.ndarray = lam0.copy()
    iterations: int = 0
    for i in range(max_iter):
        r: np.ndarray = residuals(lam)
        J: np.ndarray = jacobian(lam)
        delta: np.ndarray = np.linalg.solve(J.T @ J, -J.T @ r)

        if damping:
            for p in range(pmax):
                cost_new = cost(lam + delta / 2**p)
                cost_old = cost(lam)
                print(f"  p={p}: cost_new={cost_new}, cost_old={cost_old}")
                if cost_new < cost_old:

                # if cost(lam + delta / 2**p) < cost(lam):
                    alpha = 1 / 2**p
                    break
                if p == pmax - 1:
                    print(f"  Warnung: Kein Dämpfungsschritt gefunden nach {pmax} Versuchen")
            print(f"  Dämpfung: alpha={alpha:.4f}")
        else:
            alpha = 1.0
        
        lam = lam + alpha * delta
        iterations += 1
        print(f"Iter {i+1}: lambda={lam}, |delta|={np.linalg.norm(delta, ord=norm):.2e}")
        if np.linalg.norm(delta, ord=norm) < tol:
            print("Konvergiert!")
            break
    return lam, iterations


# Example usage:
if __name__ == "__main__":
    A0, f0, c0, f = sp.symbols('A0 f0 c0 f')
    A_data = np.array([47, 114, 223, 81, 20])
    f_data = np.array([25, 35, 45, 55, 65])

    model = A0 / ((f**2 - f0**2)**2 + c0**2)
    parameters = sp.Matrix([A0, f0, c0])
    initial_guess1 = np.array([10**8, 50, 600])
    initial_guess2 = np.array([10**7, 35, 350])
    
    best_fit, iterations = gauss_newton(model, parameters, f, f_data, A_data, initial_guess2, damping=True, norm=2)

    print(f"Beste Fit-Parameter: {best_fit}, Iterationen: {iterations}")

    f_fit = np.linspace(min(f_data), max(f_data), 5)
    A_fit = best_fit[0] / ((f_fit**2 - best_fit[1]**2)**2 + best_fit[2]**2)

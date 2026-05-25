import sympy as sp
import numpy as np

def gauss_newton_damped(
    x: np.ndarray,
    y: np.ndarray,
    f,
    initial_guess: np.ndarray,
    p,
    max_iterations: int = 100,
    pmax: int = 100,
    tolerance: float = 1e-6,
    damping: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Führt das gedämpfte Gauss-Newton-Verfahren durch, um die Koeffizienten einer Ansatzfunktion zu fitten.

    Parameters:
    x (np.ndarray): Unabhängige Variable (z.B. Anfangstemperatur, Gasdruck)
    y (np.ndarray): Abhängige Variable (z.B. Masse der entwichenen Kohlenwasserstoff-Dämpfe)
    f: Funktion, die die Form der Ansatzfunktion definiert (z.B. lambda T_Tank, T_Benzin, p_Tank, p_Benzin: ...)
    initial_guess (np.ndarray): Anfangsschätzung für die Koeffizienten
    max_iterations (int): Maximale Anzahl an Iterationen
    pmax (int): Maximale Anzahl an Dämpfungsstufen
    tolerance (float): Toleranz für die Konvergenz
    damping (bool): Flag, um die Dämpfung zu aktivieren oder zu deaktivieren

    Returns:
    np.ndarray: Koeffizienten der Ansatzfunktion
    int: Anzahl der durchgeführten Iterationen
    """
    lam = initial_guess

    g = sp.Matrix([y[k] - f(x[k], p) for k in range(len(x))])
    Dg = g.jacobian(p)

    g_lam = sp.lambdify([p], g, "numpy")
    Dg_lam = sp.lambdify([p], Dg, "numpy")

    k = 0
    increment = tolerance + 1
    err_func = np.linalg.norm(g_lam(lam)) ** 2

    while increment > tolerance and k < max_iterations:
        [Q, R] = np.linalg.qr(Dg_lam(lam))
        delta = np.linalg.solve(
            R, -Q.T @ g_lam(lam)
        ).flatten()  # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder eine "flachen" Vektor zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann

        p_damp = 0
        if damping:
            while p_damp < pmax and np.linalg.norm(g_lam(lam + delta)) ** 2 >= err_func:
                delta = delta / (2**p_damp)
                p_damp += 1

        # Update des Vektors Lambda
        lam = lam + delta
        err_func = np.linalg.norm(g_lam(lam)) ** 2
        increment = np.linalg.norm(delta)
        k = k + 1
        print("Iteration: ", k)
        print("lambda = ", lam)
        print("Inkrement = ", increment)
        print("Fehlerfunktional =", err_func)
    return lam, k

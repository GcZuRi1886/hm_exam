import numpy as np

def cubical_splinefunction(x: np.ndarray, y: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Berechnet die Werte der kubischen Splinefunktion an den Punkten in xx.
    Args:
        x (np.ndarray): Die Stützstellen der Splinefunktion.
        y (np.ndarray): Die Funktionswerte an den Stützstellen.
        xx (np.ndarray): Die Punkte, an denen die Splinefunktion ausgewertet werden soll.
    Returns:
        np.ndarray: Die Werte der Splinefunktion an den Punkten in xx.
    """
    n = len(x) - 1
    if not all(x[0] <= val <= x[n] for val in xx):
        raise ValueError("Alle Werte in xx müssen in x enthalten sein.")

    h = np.diff(x)
    a = y.copy()
    c = calculate_ci(h, y)
    b = calculate_bi(h, y, c)
    d = calculate_di(h, c)
    spline_values = np.zeros_like(xx)

    print("h:", h)
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)

    for i in range(n):
        mask = (xx >= x[i]) & (xx <= x[i + 1])
        spline_values[mask] = a[i] + b[i] * (xx[mask] - x[i]) + c[i] * (xx[mask] - x[i]) ** 2 + d[i] * (xx[mask] - x[i]) ** 3
    return spline_values

def calculate_ci(h: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Berechnet die Koeffizienten c_i für die kubische Splinefunktion.
    Args:
        h (np.ndarray): Die Abstände zwischen den Stützstellen.
        y (np.ndarray): Die Funktionswerte an den Stützstellen.
    Returns:
        np.ndarray: Die Koeffizienten c_i für die kubische Splinefunktion.
    """
    n = len(h)
    A = np.zeros((n - 1, n - 1))
    z = np.zeros(n - 1)
    c = np.zeros(n + 1)
    for i in range(1, n):
        A[i - 1, i - 1] = 2 * (h[i - 1] + h[i])
        if i > 1:
            A[i - 1, i - 2] = h[i - 1]
        if i < n - 1:
            A[i - 1, i] = h[i]
        z[i - 1] = (3 / h[i]) * (a[i + 1] - a[i]) - (3 / h[i - 1]) * (a[i] - a[i - 1])

    print("Matrix A:\n", A)
    print("Vektor z:\n", z)
    c[1:n] = np.linalg.solve(A, z)
    return c
    
def calculate_bi(h: np.ndarray, y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Berechnet die Koeffizienten b_i für die kubische Splinefunktion.
    Args:
        h (np.ndarray): Die Abstände zwischen den Stützstellen.
        y (np.ndarray): Die Funktionswerte an den Stützstellen.
        c (np.ndarray): Die Koeffizienten c_i für die kubische Splinefunktion.
    Returns:
        np.ndarray: Die Koeffizienten b_i für die kubische Splinefunktion.
    """
    n = len(h)
    b = np.zeros(n)
    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
    return b

def calculate_di(h: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Berechnet die Koeffizienten d_i für die kubische Splinefunktion.
    Args:
        h (np.ndarray): Die Abstände zwischen den Stützstellen.
        c (np.ndarray): Die Koeffizienten c_i für die kubische Splinefunktion.
    Returns:
        np.ndarray: Die Koeffizienten d_i für die kubische Splinefunktion.
    """
    n = len(h)
    d = np.zeros(n)
    for i in range(n):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    return d

if __name__ == "__main__":
    x = np.array([0, 2, 6])
    y = np.array([0.1, 0.9, 0.1])
    xx = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
    print(cubical_splinefunction(x, y, xx))

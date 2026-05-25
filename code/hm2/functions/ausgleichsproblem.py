import numpy as np


def solve_system_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Löst das lineare Gleichungssystem Ax = b unter Verwendung der QR-Zerlegung.

    Parameters:
    A (np.ndarray): Eine m x n Matrix.
    b (np.ndarray): Ein m-dimensionaler Vektor.

    Returns:
    np.ndarray: Ein n-dimensionaler Vektor, der die Lösung des Gleichungssystems darstellt.
    """
    # QR-Zerlegung von A
    Q, R = np.linalg.qr(A)
    
    # Berechnen von Q^T b
    Q_transpose_b = Q.T @ b
    
    # Lösen von R x = Q^T b durch Rückwärtssubstitution
    x = np.zeros(R.shape[1])
    for i in range(R.shape[1] - 1, -1, -1):
        x[i] = (Q_transpose_b[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    
    return x

def allgemeines_ausgleichsproblem(x: np.ndarray, y: np.ndarray, f: np.ndarray) -> (np.ndarray, np.ndarray): # type: ignore
    """
    Berechnet die Koeffizienten eines Polynoms, das die Datenpunkte (x, y) am besten approximiert, unter Verwendung der Methode der kleinsten Quadrate.
    
    Parameters:
    x (np.ndarray): Ein Array von x-Werten der Datenpunkte.
    y (np.ndarray): Ein Array von y-Werten der Datenpunkte.
    f (np.ndarray): Ein Array von Funktionswerten, die die Form des Polynoms bestimmen (z.B. [1, x, x^2] für ein quadratisches Polynom).

    Returns:
    tuple: Ein Tuple bestehend aus zwei np.ndarray:
        - coefficients: Ein Array von Koeffizienten des approximierenden Polynoms, berechnet mit der Normalengleichung.
        - coefficients_qr: Ein Array von Koeffizienten des approximierenden Polynoms, berechnet mit der QR-Zerlegung.
    """


    # Anzahl der Datenpunkte
    n = len(x)
    
    # Anzahl der Koeffizienten (Grad des Polynoms + 1)
    m = len(f)
    
    if n != len(y):
        raise ValueError("Die Länge von x und y muss gleich sein.")
    if m > n:
        raise ValueError("Die Anzahl der Koeffizienten darf nicht größer als die Anzahl der Datenpunkte sein.")

    # Erstellen der Design-Matrix A
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            A[i, j] = f[j](x[i])

    # Berechnen der Koeffizienten mit der Normalengleichung: (A^T A) c = A^T y
    A_transpose = A.T
    A_transpose_A = A_transpose @ A
    A_transpose_y = A_transpose @ y
    # Lösen des linearen Gleichungssystems
    coefficients = np.linalg.solve(A_transpose_A, A_transpose_y)

    coefficients_qr = solve_system_qr(A, y)

    return coefficients, coefficients_qr

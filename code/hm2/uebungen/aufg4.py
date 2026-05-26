import numpy as np
import matplotlib.pyplot as plt


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


def plot_solutions(results: list[tuple[np.ndarray, str]]) -> None:
    plt.figure(figsize=(10, 6))
    for values, label in results:
        plt.plot(values[:, 0], values[:, 1], label=label, marker="o")
    plt.title("Numerische Loesungen der DGL")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

def fehler_funkionale(coefficients: np.ndarray, x: np.ndarray, y: np.ndarray, f: np.ndarray) -> float:
    """
    Berechnet das Fehlerfunktional, das die Summe der quadrierten Abweichungen zwischen den Datenpunkten (x, y) und den Werten des approximierenden Polynoms darstellt.

    Parameters:
    coefficients (np.ndarray): Ein Array von Koeffizienten des approximierenden Polynoms.
    x (np.ndarray): Ein Array von x-Werten der Datenpunkte.
    y (np.ndarray): Ein Array von y-Werten der Datenpunkte.
    f (np.ndarray): Ein Array von Funktionswerten, die die Form des Polynoms bestimmen.

    Returns:
    float: Der Wert des Fehlerfunktionals.
    """
    error = 0.0
    for i in range(len(x)):
        # Berechnen des approximierten y-Werts für den aktuellen x-Wert
        y_approx = sum(coefficients[j] * f[j](x[i]) for j in range(len(coefficients)))
        # Berechnen der quadrierten Abweichung und Hinzufügen zum Fehler
        error += (y[i] - y_approx) ** 2
    return error

if __name__ == "__main__":
    x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    y = np.array([76, 92, 106, 123, 137, 151, 179, 203, 227, 250, 281, 309])
    p1 = np.array([lambda x: 1, lambda x: x, lambda x: x**2, lambda x: x**3], dtype=object)
    p2 = np.array([lambda x: 1, lambda x: x, lambda x: x**2], dtype=object)
    coefficients, coefficients_qr = allgemeines_ausgleichsproblem(x, y, p1)
    coefficients2, coefficients_qr2 = allgemeines_ausgleichsproblem(x, y, p2)
    print("Koeffizienten (Normalengleichung):", coefficients)
    print("Koeffizienten (QR-Zerlegung):", coefficients_qr)
    print("Koeffizienten 2 (Normalengleichung):", coefficients2)
    print("Koeffizienten 2 (QR-Zerlegung):", coefficients_qr2)

    error1 = fehler_funkionale(coefficients, x, y, p1)
    error2 = fehler_funkionale(coefficients2, x, y, p2)
    print("Fehlerfunktional für kubischen Fit:", error1)
    print("Fehlerfunktional für quadratischen Fit:", error2)

    # plot_solutions([
    #     (np.array([[x[i], coefficients[0] + coefficients[1] * x[i] + coefficients[2] * x[i]**2 + coefficients[3] * x[i]**3] for i in range(len(x))]), "Kubischer Fit"),
    #     (np.array([[x[i], coefficients2[0] + coefficients2[1] * x[i] + coefficients2[2] * x[i]**2] for i in range(len(x))]), "Quadratischer Fit"),
    #     (np.array(list(zip(x, y))), "Datenpunkte")
    # ])

    

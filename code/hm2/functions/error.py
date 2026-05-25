import numpy as np

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


import numpy as np
import matplotlib.pyplot as plt

def plot_halblogarithmisch(
    t: np.ndarray, N: np.ndarray, t_fit: np.ndarray, N_fit: np.ndarray
):
    """
    Plottet die Transistoranzahl in halblogarithmischer Darstellung.

    Parameters:
    t (np.ndarray): Jahre der Messwerte
    N (np.ndarray): Anzahl Transistoren (Messwerte)
    t_fit (np.ndarray): Jahre für Fitfunktion
    N_fit (np.ndarray): Anzahl Transistoren (Fit)
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(t, N, "ro", label="Messwerte", markersize=8)
    plt.semilogy(t_fit, N_fit, "b-", label="Linearer Fit", linewidth=2)
    plt.xlabel("Jahr")
    plt.ylabel("Anzahl Transistoren")
    plt.title("Moore's Law: Transistoranzahl pro Chip")
    plt.legend()
    plt.grid()
    plt.show()


def plot_polynomial_fit(x: np.ndarray, y: np.ndarray, coefficients: np.ndarray):
    """
    Plottet die Datenpunkte (x, y) und das approximierende Polynom, das durch die gegebenen Koeffizienten definiert ist.

    Parameters:
    x (np.ndarray): Ein Array von x-Werten der Datenpunkte.
    y (np.ndarray): Ein Array von y-Werten der Datenpunkte.
    coefficients (np.ndarray): Ein Array von Koeffizienten des approximierenden Polynoms.
    """
    # Erstellen eines feinen Rasters für x-Werte zum Plotten des Polynoms
    x_fit = np.linspace(min(x), max(x), 100)
    
    # Berechnen der y-Werte des Polynoms für die x_fit Werte
    y_fit = np.zeros_like(x_fit)
    for i in range(len(coefficients)):
        y_fit += coefficients[i] * (x_fit ** i)

    # Plotten der Datenpunkte und des Polynoms
    plt.scatter(x, y, color='red', label='Datenpunkte')
    plt.plot(x_fit, y_fit, color='blue', label='Approximierendes Polynom')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomielle Approximation der Datenpunkte')
    plt.legend()
    plt.grid()
    plt.show()

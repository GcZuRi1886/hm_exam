import numpy as np
import matplotlib.pyplot as plt


def draw_spline(x: np.ndarray, y: np.ndarray, xx: np.ndarray, spline_values: np.ndarray):
    """Zeichnet die kubische Splinefunktion.
    Args:
        x (np.ndarray): Die Stützstellen der Splinefunktion.
        y (np.ndarray): Die Funktionswerte an den Stützstellen.
        c (np.ndarray): Die Koeffizienten c_i für die kubische Splinefunktion.
        b (np.ndarray): Die Koeffizienten b_i für die kubische Splinefunktion.
        d (np.ndarray): Die Koeffizienten d_i für die kubische Splinefunktion.
    """
    plt.plot(x, y, 'ro', label='Stützstellen')
    plt.plot(xx, spline_values, 'b-', label='Kubische Spline')
    plt.legend()
    plt.title('Kubische Splineinterpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def draw_multiple_splines(x: np.ndarray, y: np.ndarray, xx: np.ndarray, spline_values_list: list):
    """Zeichnet mehrere kubische Splinefunktionen.
    Args:
        x (np.ndarray): Die Stützstellen der Splinefunktion.
        y (np.ndarray): Die Funktionswerte an den Stützstellen.
        c (np.ndarray): Die Koeffizienten c_i für die kubische Splinefunktion.
        b (np.ndarray): Die Koeffizienten b_i für die kubische Splinefunktion.
        d (np.ndarray): Die Koeffizienten d_i für die kubische Splinefunktion.
    """
    plt.plot(x, y, 'ro', label='Stützstellen')
    for i, spline_values in enumerate(spline_values_list):
        plt.plot(xx, spline_values, label=f'Kubische Spline {i + 1}')
    plt.legend()
    plt.title('Kubische Splineinterpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

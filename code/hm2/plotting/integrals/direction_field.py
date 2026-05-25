import numpy as np
import matplotlib.pyplot as plt

def plot_direction_field(X: np.ndarray, Y: np.ndarray, yDiff: np.ndarray, xOnes: np.ndarray):
    """Plottet das Richtungsfeld für ein gegebenes Gitter von Punkten (X, Y) und die entsprechenden Richtungsvektoren (xOnes, yDiff).
    Args:
        X (np.ndarray): Ein 2D-Array von x-Koordinaten der Gitterpunkte.
        Y (np.ndarray): Ein 2D-Array von y-Koordinaten der Gitterpunkte.
        yDiff (np.ndarray): Ein 2D-Array von y-Differenzen, die die Richtung der Vektoren in y-Richtung angeben.
        xOnes (np.ndarray): Ein 2D-Array von Einsen, das die Richtung der Vektoren in x-Richtung angibt.
    """
    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, xOnes, yDiff)
    plt.title('Direction Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

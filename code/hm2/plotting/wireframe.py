import matplotlib.pyplot as plt

def plot_wireframe(
    X, Y, Z, xlabel, ylabel, zlabel, title, figsize=(10, 8), rstride=5, cstride=5
):
    """
    Zeichnet einen 3D-Wireframe-Plot.

    Parameter:
    ----------
    X, Y, Z : ndarray
        Meshgrid-Daten für die drei Achsen
    xlabel, ylabel, zlabel : str
        Achsenbeschriftungen
    title : str
        Titel des Plots
    figsize : tuple
        Grösse der Abbildung (default: (10, 8))
    rstride, cstride : int
        Schrittweite für Zeilen/Spalten im Wireframe (default: 5)

    Returns:
    --------
    fig, ax : Figure, Axes3D
        Die erstellte Figur und Achse
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, Z, rstride=rstride, cstride=cstride)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax

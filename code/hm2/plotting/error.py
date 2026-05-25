import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def plot_global_errors(results: list[tuple[np.ndarray, str]], y_exact: Callable) -> None:
    plt.figure(figsize=(10, 6))
    for values, label in results:
        errors = np.abs(y_exact(values[:, 0]) - values[:, 1])
        plt.plot(values[:, 0], errors, label=label, marker="o")
    plt.title("Globaler Fehler |y(xi) - yi|")
    plt.xlabel("x")
    plt.ylabel("Fehler")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

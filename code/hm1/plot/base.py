import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

def f(x):
    return x 

def plot(x: np.ndarray):
    y = f(x)

    plt.plot(x, y, label="foo")
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x = np.linspace(-10, 10)
    plot(x)

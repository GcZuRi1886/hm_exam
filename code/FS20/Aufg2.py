import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

def f(x):
    return x * np.exp(x)

def df(x):
    return (x+1) * np.exp(x)

def get_cond(x):
    return np.abs((x * df(x))/f(x))

def plot_cond(x: np.ndarray):
    cond = get_cond(x)

    plt.plot(x, cond)
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x = np.linspace(-10, 10)
    plot_cond(x)

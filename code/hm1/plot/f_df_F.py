import numpy as np
import matplotlib.pyplot as plt

def f(coeffs: list[int], x: np.ndarray):
    return sum(c * x**i for i, c in enumerate(coeffs))

def df(coeffs: list[int], x: np.ndarray):
    return sum(i * c * x**(i - 1) for i, c in enumerate(coeffs) if i > 0)

def F(coeffs: list[int], x: np.ndarray):
    return np.sum([c / (i + 1) * x**(i + 1) for i, c in enumerate(coeffs)], axis=0)

def plot_function(coeffs: list[int], x: np.ndarray):
    y = f(coeffs, x)
    dy = df(coeffs, x)
    Y = F(coeffs, x)

    plt.plot(x, y, label='f(x)', color='blue')
    plt.plot(x, dy, label="f'(x)", color='orange')
    plt.plot(x, Y, label='F(x)', color='green')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.title('Plot f(x), seine Ableitung und Stammfunktion')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim(-2500, 2000)
    plt.xlim(-10, 10)
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    coeffs = [ 213, 123, 33, 56 ]
    plot_function(coeffs, x)


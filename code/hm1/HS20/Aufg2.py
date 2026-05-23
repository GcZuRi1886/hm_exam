import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

def f(x):
    return x**2 * np.sin(x)

def df(x):
    return 2*x * np.sin(x) + x**2 * np.cos(x)


def get_cond(x):
    return np.abs((x * df(x))/f(x))

#b)
# Maximaler relativer fehler 10%
def ex_b(x_0, max_err):
    rel_err = 0.1
    abs_err = rel_err/get_cond(x_0) * x_0
    print(abs_err)

#c)
def lim_0_cond():
    print([get_cond(10**-x) for x in range(10, 20)])

#d)
def plot(x: np.ndarray):
    plt.semilogy(x, get_cond(x), label="")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # print(get_cond(2))$
    # lim_0_cond()
    x = np.linspace(-2*np.pi, 3*np.pi)
    plot(x)

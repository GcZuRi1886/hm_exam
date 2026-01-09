import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

def f(x):
    return 1 / (np.cos(x + (np.pi / 4)) -1) + 2

def df(x):
    return (-1 * np.sin(x + np.pi/4))/(np.cos(x+np.pi/4)-1)**2

def plot(x: np.ndarray):
    plt.plot(x, f(x), color="blue", label="f")
    plt.plot(x, df(x), color="red", label="df")
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()


# b 
# War erfolgreich mit alpha = 0.9 (Leibschitzkonstante
def banach(x):
    min = np.min(x)
    max =  np.max(x)

    alpha = np.abs((f(max) - f(min)) / (max - min))
    if 0 < alpha < 1:
        print(f"Successful with alpha: {alpha}")
    

# c
def a_posteriori(alpha, x_n, x_n_1):
    return alpha / (1 - alpha) * np.abs(x_n - x_n_1)

def fixpoint_iteration(f, x_0, tolerance, alpha):
    error = np.inf
    fixpoint = 0
    x = x_0
    iteration_count = 0
    while error > tolerance:
        fixpoint = f(x)
        error = a_posteriori(alpha, fixpoint, x)
        x = fixpoint
        iteration_count += 1

    return iteration_count, fixpoint

# d)
def g(x):
    return (x-1)/(x-2) - np.cos(x + np.pi/4)

    

if __name__ == "__main__":
    x = np.linspace(0, np.pi)
    # plot(x)
    # banach(x)
    # print(f"A posteriori: {a_posteriori(0.9, 1.3441, 1.3376)}")
    print(f"fixpoint_iteration: {fixpoint_iteration(f, 3, 0.01, 0.9)}")


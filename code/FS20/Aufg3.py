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
    

if __name__ == "__main__":
    x = np.linspace(0, np.pi)
    # plot(x)
    banach(x)


import numpy as np



def f1(x):
    return 55*x**3 + 64 * x**2


def f2(x):
    return x**5 + 1.8 * x**4 + 348 * x + 752

def secant_method(func, x0, x1, precision=1e-3):
    while True:
        x2 = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        if abs(x2 - x1) < precision:
            return x2 
        x0, x1 = x1, x2


if __name__ == "__main__":
    print(secant_method(f1, -9, 9, precision=1e-10)) 
    print(secant_method(f2, -9, 9, precision=1e-10)) 

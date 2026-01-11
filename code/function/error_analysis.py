import numpy as np

def f(x):
    return x**2

def df(x):
    return 2*x

def get_cond(x):
    return np.abs((x * df(x))/f(x))

def absolute_error(df, x, x_err):
    return np.abs(df(x)) * np.abs(x_err - x)

def relative_error(f, df, x, x_err):
    left_frac = np.abs(x * df(x) / f(x))
    right_frac = np.abs((x_err - x) / x)

    return left_frac * right_frac

import numpy as np

class ButcherTableau:
    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        self.a = a
        self.b = b
        self.c = c

class StepSizeControl:
    def __init__(self, a: float, b: float, h: float | None = None, n: int | None = None):
        if h is not None and n is not None:
            raise ValueError("Only one of h or n should be provided.")
        if h is None and n is None:
            raise ValueError("Either h or n must be provided.")
        
        if h is not None:
            self.h = h
            self.n = int((b - a) / h)
        elif n is not None:
            self.n = n
            self.h = (b - a) / n

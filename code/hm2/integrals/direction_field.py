import numpy as np
from typing import Callable


def prepare_direction_field(f: Callable, xmin: float, xmax: float, ymin: float, ymax: float, hx: float, hy: float):
    """Prepares the data for plotting a direction field for a given function f.

    Parameters:
    f (Callable): The function for which to prepare the direction field.
    xmin (float): The minimum x-value of the grid.
    xmax (float): The maximum x-value of the grid.
    ymin (float): The minimum y-value of the grid.
    ymax (float): The maximum y-value of the grid.
    hx (float): The step size in the x-direction.
    hy (float): The step size in the y-direction.

    Returns:
    tuple: A tuple containing the grid points (X, Y), the x-components of the direction field (xOnes), and the y-components of the direction field (yDiff).
    """
    x = np.arange(xmin, xmax + hx, hx)
    y = np.arange(ymin, ymax + hy, hy)
    X, Y = np.meshgrid(x, y)
    
    yDiff = f(X, Y)
    xOnes = np.ones_like(yDiff)
    
    return X, Y, xOnes, yDiff

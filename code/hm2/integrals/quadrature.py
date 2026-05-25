from typing import Callable

def sum_rechteck(
    h: float, n: int, x_list: list[float], f: Callable[[float], float]
) -> float:
    """Calculates the integral of a function f using the right rectangle method.

    Parameters:
    h (float): The width of each rectangle.
    n (int): The number of rectangles.
    x_list (list[float]): The list of x values at which to evaluate the function f.
    f (Callable[[float], float]): The function to integrate.

    Returns:
    float: The approximate integral of the function f.
    """

    integral = 0.0
    for i in range(n):
        integral += f(x_list[i] + h / 2)

    return integral * h

def sum_trapez(h: float, n: int, x_list: list[float], a: float, b: float, f: Callable[[float], float]) -> float:
    """Calculates the integral of a function f using the trapezoidal rule.

    Parameters:
    h (float): The width of each trapezoid.
    n (int): The number of trapezoids.
    x_list (list[float]): The list of x values at which to evaluate the function f.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    f (Callable[[float], float]): The function to integrate.

    Returns:
    float: The approximate integral of the function f.
    """

    integral = (f(a) + f(b)) / 2
    for i in range(1, n):
        integral += f(x_list[i])

    return integral * h

def sum_simpson(h: float, n: int, x_list: list[float], f: Callable[[float], float]) -> float:
    """Calculates the integral of a function f using Simpson's rule.

    Parameters:
    h (float): The width of each interval.
    n (int): The number of intervals (must be even).
    x_list (list[float]): The list of x values at which to evaluate the function f.
    f (Callable[[float], float]): The function to integrate.

    Returns:
    float: The approximate integral of the function f.
    """

    integral = f(x_list[0]) / 2

    for i in range(1, n):
        integral += f(x_list[i]) 

    for i in range(1, n+1):
        integral += 2 * f((x_list[i-1] + x_list[i]) / 2)

    integral += f(x_list[-1]) / 2

    return integral * h / 3

def sum_trapez_non_equi(x_list: list[float], y_list: list[float]) -> float:
    """Calculates the integral of a function f using the trapezoidal rule for non-equidistant x values.

    Parameters:
    x_list (list[float]): The list of x values
    y_list (list[float]): Already evaluated y values

    Returns:
    float: The approximate integral of the function f.
    """

    n = len(x_list) - 1
    integral = 0.0
    for i in range(n):
        integral += (y_list[i] + y_list[i+1]) / 2 * (x_list[i+1] - x_list[i])

    return integral

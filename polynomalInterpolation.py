import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Qt5Agg')


def solveLGS(A, b):
    n = len(A)
    # Vorwärtssubstitution
    for k in range(n):
        for i in range(k + 1, n):
            if A[i][k] == 0:
                continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # Rückwärtssubstitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / A[i][i]

    return x


def get_polynomial_coefficients(x_points, y_points):
    # Erstelle die Vandermonde-Matrix
    A = np.vander(x_points, increasing=True).tolist()
    b = y_points[:]
    # Löse das LGS
    coefficients = solveLGS(A, b)
    return coefficients

def get_polynomial_coefficients_polyfit(x_points, y_points):
    coefficients = np.polyfit(x_points, y_points, len(x_points) - 1)
    return list(reversed(coefficients))

def draw_polynomial(coefficients, x_range):
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.polyval(list(reversed(coefficients)), x_vals)

    plt.plot([val + 1997 for val in x_vals], y_vals, label='Interpolierendes Polynom')
    plt.scatter([point + 1997 for point in x_points], y_points, color='red', label='Datenpunkte')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynominterpolation')
    plt.legend()
    plt.grid()
    plt.show()


x_points = [0, 2, 9, 13]
y_points = [150, 104, 172, 152]

coefficients = get_polynomial_coefficients(x_points, y_points)
coefficients_polyfit = get_polynomial_coefficients_polyfit(x_points, y_points)

print("Koeffizienten des interpolierenden Polynoms:", coefficients)

draw_polynomial(coefficients, (min(x_points) - 1, max(x_points) + 1))
draw_polynomial(coefficients_polyfit, (min(x_points) - 1, max(x_points) + 1))

# Solve LGS
# 2003 --> 126
# 2004 --> 142.7
# Polyfit hat sozusagen die selben Werte für 2003 und 2004 wie mein selbstgeschriebener Algorithmus.

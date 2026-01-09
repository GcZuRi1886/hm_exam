import numpy as np

A = np.array([[15, 0, 1],
     [1, 3, 7],
     [0, 1, 6]], dtype=float)

y = np.array([67,21,44], dtype=float)


def prepare_gauss_seidel(A, b):
    """Prepares the matrix A and y vector for the Gauss-Seidel method."""
    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    R = A - D - L
    print("D:", D)
    print("L:", L)
    print("R:", R)
    DL_inv = np.linalg.inv(D + L)
    B = -DL_inv @ R # Gauss-Seidel iteration matrix
    D_inv_b = DL_inv @ b
    return B, D_inv_b

def a_posteriori_error(B, x_n, x_n_minus_1):
    """Calculates the a posteriori error estimate."""
    norm_B = np.linalg.norm(B, ord=np.inf)

    diff = x_n - x_n_minus_1
    norm_diff = np.linalg.norm(diff, ord=np.inf)
    return (norm_B / (1 - norm_B)) * norm_diff

def gauss_seidel(A, b, x, iterations=None, tolerance=None):
    """Performs recursive iterations of the Gauss-Seidel method."""
    iteration_count = 0
    if tolerance is not None:
        error = float('inf')
        B, _ = prepare_gauss_seidel(A, b)
        while error > tolerance:
            x_new = gauss_seidel_iteration(A, b, x)
            error = a_posteriori_error(B, x_new, x)
            x = x_new
            iteration_count += 1
    elif iterations is not None:
        while iterations > 0:
            x = gauss_seidel_iteration(A, b, x)
            iterations -= 1
            iteration_count += 1
    
    return x, iteration_count


def gauss_seidel_iteration(A, b, x):
    x_new = np.zeros_like(x)
    for i in range(len(x)):
        x_new[i] = 1 / A[i, i] * (b[i] - A[i, :i] @ x_new[:i] - A[i, i+1:] @ x[i+1:])
    
    return x_new

if __name__ == "__main__":
    print(gauss_seidel(A, y, [0,0,0], 5))

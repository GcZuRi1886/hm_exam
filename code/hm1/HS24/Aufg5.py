import numpy as np


def prepare_jacobi(A, b):
    """Prepares the matrix A and b vector for the Jacobi method."""
    D = np.diag(np.diag(A))
    L_plus_R = A - D
    D_inv = np.linalg.inv(D)
    B = -D_inv @ L_plus_R # Jacobi iteration matrix
    D_inv_b = D_inv @ b
    return B, D_inv_b

def prepare_jacobi_alternative(A, b, omega):
    """Prepares the matrix A and b vector for the Jacobi method."""
    D = np.diag(np.diag(A))
    L_plus_R = A - D
    D_inv = np.linalg.inv(D)
    B = (-1) * (omega * D_inv @ ( (omega - 1) / omega * D + L_plus_R))
    D_inv_b = omega * D_inv @ b
    return B, D_inv_b

def jacobi(B, D_inv_b, x, iterations=None, tolerance = None):
    """Performs recursive iterations of the Jacobi method."""
    iteration_count = 0
    if tolerance is not None:
        error= float('inf')
        while error > tolerance:
            x_new = jacobi_iteration(B, D_inv_b, x)
            error = a_posteriori_error(B, x_new,x)
            x = x_new
            iteration_count += 1
    elif iterations is not None:
        while iterations > 0:
            x = jacobi_iteration(B, D_inv_b, x)
            iterations -= 1
            iteration_count += 1
    
    return x, iteration_count

def jacobi_iteration(B, D_inv_b, x):
    """Performs recursive iterations of the Jacobi method."""
    return np.array(D_inv_b + B @ x)

def a_posteriori_error(B, x_n, x_n_minus_1):
    """Calculates the a posteriori error estimate."""
    norm_B = np.linalg.norm(B, ord=1)

    diff = x_n - x_n_minus_1
    norm_diff = np.linalg.norm(diff, ord=1)
    return (norm_B / (1 - norm_B)) * norm_diff

if __name__ == "__main__":
    A = np.array([[7, -2, -2],
                 [-2, 7, -2],
                 [-2, -2, 7]])
    b = np.array([5, -13, 14])
    x_0 = np.zeros(3)
    omega = 1.15

    B, D_inv_b = prepare_jacobi(A, b)
    B_alt, D_inv_b_alt = prepare_jacobi_alternative(A, b, omega)
    print(B_alt, D_inv_b_alt)
    tolerance = 1e-9
    x, iterations = jacobi(B, D_inv_b, x_0, tolerance=tolerance)
    x_alt, iter_alt = jacobi(B_alt, D_inv_b_alt, x_0, tolerance=tolerance)
    print(f"Jacobi converged in {iterations} iterations to solution: {x}")
    print(f"Alternative Jacobi converged in {iter_alt} iterations to solution: {x_alt}")

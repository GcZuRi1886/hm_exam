import numpy as np

def prepare_jacobi(A, b):
    """Prepares the matrix A and b vector for the Jacobi method."""
    D = np.diag(np.diag(A))
    L_plus_R = A - D
    D_inv = np.linalg.inv(D)
    B = -D_inv @ L_plus_R # Jacobi iteration matrix
    D_inv_b = D_inv @ b
    return B, D_inv_b

def jacobi(B, D_inv_b, x, iterations=None, tolerance = None):
    """Performs recursive iterations of the Jacobi method."""
    iteration_count = 0
    error= float('inf')
    if tolerance is not None:
        while error > tolerance:
            x_new = jacobi_iteration(B, D_inv_b, x)
            error = a_posteriori_error(B, x_new,x)
            x = x_new
            iteration_count += 1
    elif iterations is not None:
        while iterations > 0:
            x_new = jacobi_iteration(B, D_inv_b, x)
            error = a_posteriori_error(B, x_new,x)
            x = x_new
            iterations -= 1
            iteration_count += 1
    
    return x, iteration_count, error

def jacobi_iteration(B, D_inv_b, x):
    """Performs recursive iterations of the Jacobi method."""
    return np.array(D_inv_b + B @ x)

def a_posteriori_error(B, x_n, x_n_minus_1):
    """Calculates the a posteriori error estimate."""
    norm_B = np.linalg.norm(B, ord=np.inf)

    diff = x_n - x_n_minus_1
    norm_diff = np.linalg.norm(diff, ord=np.inf)
    return (norm_B / (1 - norm_B)) * norm_diff

def a_priori_error(B, x0, x1, iterations):
    """Calculates the a priori error estimate."""
    norm_B = np.linalg.norm(B, ord=np.inf)
    norm_diff = np.linalg.norm(x1 - x0, ord=np.inf)
    return (norm_B ** iterations / (1 - norm_B)) * norm_diff

def a_priori_iterations(B, x0, x1, tolerance):
    norm_B = np.linalg.norm(B, np.inf)

    if norm_B >= 1:
        raise ValueError("Jacobi-Verfahren konvergiert nicht (||B|| >= 1).")
    error0 = np.linalg.norm(x1 - x0, np.inf)

    # a-priori Abschätzung
    k = np.log((tolerance * (1 - norm_B)) / error0) / np.log(norm_B)


    return int(np.ceil(k))

c = 4

A = np.array([[c, -1, 0, 0, 0, 0],
             [-1, c, -1, 0, 0, 0],
             [0, -1, c, -1, 0, 0],
             [0, 0, -1, c, -1, 0],
             [0, 0, 0, -1, c, -1],
             [0, 0, 0, 0, -1, c]])

b = np.array([c, 0, 0, 0, 0, c])

x0 = np.zeros(6)

if __name__ == "__main__":
    B, D_inv_b = prepare_jacobi(A, b)
    print(B)

    iterations = a_priori_iterations(B, x0, jacobi_iteration(B, D_inv_b, x0), 1e-3)
    result, iteration_count, error = jacobi(B, D_inv_b, x0, iterations=iterations)
    print(result, iteration_count, error)

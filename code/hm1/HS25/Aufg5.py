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
    if tolerance is not None:
        error= float('inf')
        while error > tolerance:
            x_new = jacobi_iteration(B, D_inv_b, x)
            error = a_posteriori_error(B, x_new,x)
            x = x_new
            iteration_count += 1
    elif iterations is not None:
        while iterations > 0:
            x_new = jacobi_iteration(B, D_inv_b, x)
            x = x_new
            iterations -= 1
            iteration_count += 1
    
    return x, iteration_count

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

if __name__ == "__main__":
    A = np.array([[-3, 1, 0],
                 [2, 5, -1],
                 [1, 0, 3]])
    b = np.array([-7, -2, 5])
    B, c = prepare_jacobi(A, b)
    B_norm = np.linalg.norm(B, np.inf)
    print(f"B is: {B} and c is: {c}")
    print(f"Inf norm is: {B_norm}")
    """ 
    a)
    B is: 
    [[ 0.          0.33333333  0.        ]
    [-0.4         0.          0.2       ]
    [-0.33333333  0.          0.        ]]
    c is: [ 2.33333333 -0.4         1.66666667]
    
    B ist nicht diagonal dominant

    Inf Norm ist: 0.6000000000000001
    Jacobi konvergiert nicht, da nicht diagonaldominant
    """

    # b)

    x_0 = np.zeros(3)
    x, iters = jacobi(B, c, x_0, iterations=10)
    x_min_1, iters = jacobi(B, c, x_0, iterations=9)
    x_1 = jacobi_iteration(B, c, x_0)

    print(f"Die Lösung nach 10 Schritten ist: {x}")
    print(f"a priori error: {a_priori_error(B, x_0, x_1, 10)}")
    print(f"a posteriori error: {a_posteriori_error(B, x, x_min_1)}")

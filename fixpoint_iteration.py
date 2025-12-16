import numpy as np


def prepare_jacobi(A, b):
    """Prepares the matrix A and b vector for the Jacobi method."""
    D = np.diag(np.diag(A))
    L_plus_R = A - D
    D_inv = np.linalg.inv(D)
    B = -D_inv @ L_plus_R # Jacobi iteration matrix
    D_inv_b = D_inv @ b
    return B, D_inv_b

def prepare_gauss_seidel(A, b):
    """Prepares the matrix A and b vector for the Gauss-Seidel method."""
    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    R = A - D - L
    DL_inv = np.linalg.inv(D + L)
    B = -DL_inv @ R # Gauss-Seidel iteration matrix
    D_inv_b = DL_inv @ b
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

def run_function(function, matrix_calc, matrix_approx, vector, initial_guess, iterations=None, tolerance=None):
    """Runs the given function and computes error estimates."""

    x_n_approx, iteration_count = function(matrix_calc, vector, np.copy(initial_guess), iterations, tolerance)
    x_n_minus_1, _ = function(matrix_calc, vector, np.copy(initial_guess), iteration_count - 1)
    x_1_approx, _ = function(matrix_calc, vector, np.copy(initial_guess), 1)

    a_post_error = a_posteriori_error(matrix_approx, x_n_approx, x_n_minus_1)
    a_prior_error = a_priori_error(matrix_approx, initial_guess, x_1_approx, iteration_count)
    return x_n_approx, a_post_error, a_prior_error, iteration_count


def print_output(function, iteration_count = None, tolerance = None):
    if tolerance is not None:
        print(f"Approximate solution using {function.__name__} with tolerance {tolerance}")
    elif iteration_count is not None:
        print(f"Approximate solution using {function.__name__} after {iteration_count} iterations")
    else:
        print("Specify either iteration_count or tolerance")
        return

    D_inv_b = None
    if function == gauss_seidel:
        B, D_inv_b = prepare_gauss_seidel(A, b)
        print("Gauss-Seidel iteration matrix B:\n", B)
        x_n_approx, a_post_error, a_prior_error, iteration_count = run_function(gauss_seidel, A, B, b, x0, iteration_count, tolerance)
    elif function == jacobi:
        B, D_inv_b = prepare_jacobi(A, b)
        print("Jacobi iteration matrix B:\n", B)
        x_n_approx, a_post_error, a_prior_error, iteration_count = run_function(jacobi, B, B, D_inv_b, x0, iteration_count, tolerance)
    else:
        raise ValueError("Unsupported function")

    real_error = np.linalg.norm(x_n_approx - real_solution, ord=np.inf)

    print("Approximate solution ", x_n_approx)
    print("A posteriori error estimate:", a_post_error)
    print("A priori error estimate:", a_prior_error)
    print("Real error:", real_error)
    print("Number of iterations:", iteration_count)


A = np.array([[8, 5, 2],
             [5, 9, 1],
             [4, 2, 7]], dtype=float)
b = np.array([19, 5, 34], dtype=float)
x0 = np.array([1, -1, 3], dtype=float)
real_solution = np.array([2,-1,4], dtype=float)

iteration_count = 3
tolerance = 10**-5
# tolerance = None

if __name__ == "__main__":
    print_output(jacobi, iteration_count=iteration_count, tolerance=tolerance)
    print("\n")
    print_output(gauss_seidel, iteration_count=iteration_count, tolerance=tolerance)

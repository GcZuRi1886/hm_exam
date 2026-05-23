import numpy as np

def lr_zerlegung(A):
    """
    Führt eine LR-Zerlegung (A = L @ R) ohne Pivotisierung durch.
    
    Parameter:
        A : quadratische numpy-Matrix (n x n)
    
    Rückgabe:
        L : untere Dreiecksmatrix mit Einsen auf der Diagonale
        R : obere Dreiecksmatrix
    """
    A = A.astype(float)          # wichtig für Divisionen
    n = A.shape[0]

    L = np.eye(n)
    R = A.copy()

    for k in range(n - 1):
        if R[k, k] == 0:
            raise ValueError("LR-Zerlegung ohne Pivotisierung nicht möglich (Null-Pivot).")

        for i in range(k + 1, n):
            L[i, k] = R[i, k] / R[k, k]
            R[i, k:] = R[i, k:] - L[i, k] * R[k, k:]

    return L, R


def sub_a(A, b):
    L, R = lr_zerlegung(A)
    print("L:\n", L)
    print("R:\n", R)
    # Lösen von Ly = b
    y = np.linalg.solve(L, b)
    # Lösen von Rx = y
    x = np.linalg.solve(R, y)
    print("Lösung x:\n", x)



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
            x = jacobi_iteration(B, D_inv_b, x)
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

if __name__ == "__main__":
    A_m = np.array([[4, 1, 0],
                    [1, 4, 1],
                    [0, 1, 4]])
    b = np.array([1, 2, 3])
    
    sub_a(A_m, b)
    B, D_inv_b = prepare_jacobi(A_m, b)
    x0 = np.zeros_like(b)
    x_jacobi, iters = jacobi(B, D_inv_b, x0, tolerance=1e-8)
    print("Jacobi Lösung x:\n", x_jacobi)
    print("Jacobi Iterationen:", iters)

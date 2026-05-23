import numpy as np

def von_mises_iteration(A, num_iterations=100, tolerance=1e-10, start_vector=None):
    """
    Calculate the dominant eigenvalue and corresponding eigenvector of matrix A
    using the Von Mises iteration method.

    Parameters:
    A (np.ndarray): A square matrix.
    num_iterations (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    float: Dominant eigenvalue.
    np.ndarray: Corresponding eigenvector.
    int: Number of iterations performed.
    """
    n = A.shape[0]
    
    if start_vector is not None:
        b_k = start_vector / np.linalg.norm(start_vector)
    else:
        b_k = np.random.rand(n)
    iterations_done = 0
    
    for _ in range(num_iterations):
        iterations_done += 1

        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        
        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        
        # Re-normalize the vector
        b_k1 = b_k1 / b_k1_norm
        
        # Check for convergence
        if np.linalg.norm(b_k1 - b_k) < tolerance:
            break
        
        b_k = b_k1
    
    # Rayleigh quotient to estimate the dominant eigenvalue
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
    
    return eigenvalue, b_k, iterations_done

def characteristic_polynomial(A: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    """
    Berechnet die Koeffizienten des charakteristischen Polynoms von A:
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A muss eine quadratische 2D-Matrix sein.")

    eigvals = np.linalg.eigvals(A)

    coeffs = np.poly(eigvals)

    if np.max(np.abs(np.imag(coeffs))) < tol:
        coeffs = np.real(coeffs)

    coeffs[np.abs(coeffs) < tol] = 0.0
    return coeffs

# Example usage
if __name__ == "__main__":
    c = -3
    A = np.array([[30, c],
                  [-13, 4]], dtype=float)
    v_0 = np.array([1, 0])

    eigenvalue, eigenvector, iterations = von_mises_iteration(A, tolerance=1e-15, start_vector=v_0)
    print("Dominant Eigenvalue:", eigenvalue)
    print("Corresponding Eigenvector:", eigenvector)
    print("Iterations performed:", iterations)

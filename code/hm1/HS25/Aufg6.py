import numpy as np


def qr_householder(A):
    # Berechnet QR-Zerlegung der nxn-Matrix A mit Householder-Verfahren

    A = np.copy(A)
    n = np.shape(A)[0]

    R = A
    Q = np.eye(n)

    for i in np.arange(0,n-1):
        a = R[i:n,i]
        e = np.zeros(n-i)
        e[0] = 1
        if a[0] >= 0:
            sig = -1
        else:
            sig = 1 
        v = a - sig * np.linalg.norm(a) * e
        u = v / np.linalg.norm(v)
        E = np.eye(n-i)
        u = u.reshape(n-i,1)
        H = E - 2 * (u @ u.T)
        Qi = np.eye(n)
        Qi[i:n,i:n] = H
        R = Qi @ R
        Q = Q @ Qi.T

    return Q, R


def qr_eigenvalues(A, num_iterations=100):
    # Berechnet Eigenwerte der nxn-Matrix A mit QR-Algorithmus

    A = np.copy(A)
    P = np.eye(A.shape[0])

    for _ in range(num_iterations):
        Q, R = qr_householder(A)
        A = R @ Q
        P = P @ Q

    eigenvalues = np.diag(A)
    return eigenvalues

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

def diagonalize_matrix(matrix, tMatrix):
    """
    Diagonalize a given square matrix using a transformation matrix.

    Parameters:
    matrix (np.ndarray): A square matrix to diagonalize.
    tMatrix (np.ndarray): The transformation matrix.

    Returns:
    np.ndarray: The diagonalized matrix.
    """
    if not isinstance(matrix, np.ndarray) or not isinstance(tMatrix, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be a square matrix.")
    
    if tMatrix.ndim != 2 or tMatrix.shape[0] != tMatrix.shape[1]:
        raise ValueError("Transformation matrix must be a square matrix.")
    
    if matrix.shape[0] != tMatrix.shape[0]:
        raise ValueError("Matrix and transformation matrix must have the same dimensions.")
    
    try:
        tMatrix_inv = np.linalg.inv(tMatrix)
    except np.linalg.LinAlgError:
        raise ValueError("Transformation matrix is not invertible.")
    
    diagonalized = tMatrix_inv @ matrix @ tMatrix
    return diagonalized

if __name__ == "__main__":
    # a)
    A = np.array([[-5, 4, 8],
                 [8, -1, -10],
                 [-8, 4, 11]])
    eigenvalues = qr_eigenvalues(A, num_iterations=1000)
    print(f"Die Eigenwerte sind: {eigenvalues}")

    # b)
    eigen_np = np.linalg.eig(A)
    print(f"Die Eigenwerte von Numpy sind: {eigen_np.eigenvalues}")

    # D = T^-1 A T
    t = eigen_np.eigenvectors
    print(f"Diagonal Matrix: {diagonalize_matrix(A, t)}")
    
    """Diagonal Matrix:
    [[ 1.00000000e+00+2.00000000e+00j  9.82547377e-15-4.21884749e-15j 1.43890180e-15-1.35119861e-14j]
    [  7.32747196e-15+3.77475828e-15j  1.00000000e+00-2.00000000e+00j -1.74668301e-16+1.34429669e-14j]
    [  3.91316838e-15-2.49777278e-15j  4.69821461e-15+3.28281901e-15j 3.00000000e+00+6.66133815e-16j]]
    """


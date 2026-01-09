import numpy as np

def calculate_eigenvectors_and_eigenvalues(matrix, tMatrix=None):
    """
    Calculate the eigenvectors and the eigenvalues of a given square matrix.
    
    Parameters:
    matrix (np.ndarray): A square matrix to calculate eigenvectors for.
    tMatrix (np.ndarray, optional): An optional transformation matrix.
    
    Returns:
    np.ndarray: The eigenvectors of the matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    if tMatrix is not None:
        if not isinstance(tMatrix, np.ndarray):
            raise ValueError("Transformation matrix must be a numpy array.")
        
        if tMatrix.ndim != 2 or tMatrix.shape[0] != tMatrix.shape[1]:
            raise ValueError("Transformation matrix must be a square matrix.")
        
        # Apply the transformation
        transformed_matrix = np.linalg.inv(tMatrix) @ matrix @ tMatrix
        eigenvalues, eigenvectors = np.linalg.eig(transformed_matrix)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    return eigenvalues, eigenvectors


def qr_eigenvalues(A, num_iterations=100):
    # Berechnet Eigenwerte der nxn-Matrix A mit QR-Algorithmus

    A = np.copy(A)
    P = np.eye(A.shape[0])

    for _ in range(num_iterations):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        P = P @ Q

    if(not np.allclose(P.T @ P, np.eye(5))):
        print("Warning: Q is not orthogonal")
    else:
        print("Q is orthogonal")

    eigenvalues = np.diag(A)
    return eigenvalues


if __name__ == "__main__":
    A = np.array([[6,1,2,1,2],
              [1,5,0,2,-1],
              [2,0,5,-1,0],
              [1,2,-1,6,1],
              [2,-1,0,1,7]], dtype=float)


    eigenvalues, eigenvectors = calculate_eigenvectors_and_eigenvalues(A)
    print(f"Eigenvalues:\n{eigenvalues}\n")
    print("Eigenvectors:")
    for vec in eigenvectors.T:
        print(vec)
    print()
    eigenvalues_qr = qr_eigenvalues(A)
    print(f"Eigenvalues from QR algorithm:\n{eigenvalues_qr}\n")

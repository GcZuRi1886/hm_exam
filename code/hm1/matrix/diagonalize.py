import numpy as np

def check_diagonalizable(matrix):
    """
    Check if a given square matrix is diagonalizable.

    Parameters:
    matrix (np.ndarray): A square matrix to check.

    Returns:
    bool: True if the matrix is diagonalizable, False otherwise.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Count the number of linearly independent eigenvectors
    rank = np.linalg.matrix_rank(eigenvectors)
    
    return rank == matrix.shape[0]

def check_diagonalizable_manual(matrix, tMatrix):
    """
    Check if a given square matrix is diagonalizable using a transformation matrix.

    Parameters:
    matrix (np.ndarray): A square matrix to check.
    tMatrix (np.ndarray): The transformation matrix.

    Returns:
    bool: True if the matrix is diagonalizable, False otherwise.
    """
    
    diagonalized = diagonalize_matrix(matrix, tMatrix)
    is_diagonal = np.allclose(diagonalized, np.diag(np.diag(diagonalized)))
    
    return is_diagonal

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


# Example usage
if __name__ == "__main__":
    A = np.array([[2, 0, 1],
                  [7, -5, 9],
                  [6, -6, 9]])
    
    T = np.array([[3, 1, 1],
                  [-1, 1, 2],
                  [-3, 0, 1]])

    print("Is the matrix diagonalizable (using eigenvectors)?", check_diagonalizable(A))
    print("Is the matrix diagonalizable (using transformation matrix)?", check_diagonalizable_manual(A, T))
    print("Diagonalized matrix:\n", diagonalize_matrix(A, T))

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

if __name__ == "__main__":
    A = np.array([[3, 1],
                  [1, 3]])
    eigenvalues, eigenvectors = calculate_eigenvectors_and_eigenvalues(A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

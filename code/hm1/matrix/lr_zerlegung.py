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


A_m = np.array([[-1, 1, 1],
               [1, -3, -2],
               [5, 1, 4]])

print(lr_zerlegung(A_m))

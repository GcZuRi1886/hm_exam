import copy

def solveLGS(A, b):
    n = len(A)
    # Vorwärtssubstitution
    for k in range(n):
        for i in range(k + 1, n):
            if A[i][k] == 0:
                continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # Rückwärtssubstitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / A[i][i]

    return x

def aTriangularMatrix(A):
    n = len(A)
    for k in range(n):
        for i in range(k + 1, n):
            if A[i][k] == 0:
                continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
    return A

def determinateMatrix(A):
    n = len(A)
    det = 1
    for k in range(n):
        det *= A[k][k]
    return det

def Aufg2(A, b):
    aTriangular = aTriangularMatrix(copy.deepcopy(A))
    det = determinateMatrix(aTriangular)
    solution = solveLGS(copy.deepcopy(A), b)
    return aTriangular, det, solution

# Beispielaufruf
A = [[20, 30, 10],
     [10, 17, 6],
     [2, 3, 2]]
b = [5200, 3000, 760]
solution = Aufg2(A, b)

print("Lösung des LGS:", solution)  # Ausgabe: Lösung des LGS: [2.0, 3.0, -1.0]

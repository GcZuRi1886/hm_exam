import numpy as np

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
        print(f"{A}={b}")


    # Rückwärtssubstitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / A[i][i]

    return x

def max_relative_error_vector(b, b_err, A_cond):
    b_norm = np.linalg.norm(b, np.inf)
    b_err_norm = np.linalg.norm(b - b_err, np.inf)

    return A_cond * b_err_norm / b_norm

if __name__ == "__main__":
    A = np.array([[240, 120, 80],
                  [60, 180, 170],
                  [60, 90, 500]], dtype=float)
    b = np.array([3080,4070,5030], dtype=float)
    b_err = b * 0.05

    print(b_err)

    A_cond = np.linalg.norm(A, np.inf) * np.linalg.norm(np.linalg.inv(A), np.inf)

    print(A_cond)

    # print(solveLGS(A, b))
    print(max_relative_error_vector(b, b_err, A_cond))

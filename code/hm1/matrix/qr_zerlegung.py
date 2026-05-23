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

A = np.array(
        [
            [1, -2, 3],
            [-5, 4, 1],
            [2, -1, 3]
        ]
)
Q, R = qr_householder(A)
print(f"QR-Zerlegung von A:\nQ =\n{Q}\nR =\n{R}\n")

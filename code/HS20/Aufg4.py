import numpy as np

e = 2**(-52)

L = np.array([[1, 0, 0],
             [2, 1, 0],
             [4, 2/e, 1]])

R = np.array([[1, 1, 1],
             [0, e, 3],
             [0, 0, 4 - 6/e]])

A = np.array([[1, 1, 1],
             [2, 2+e, 5],
             [4, 6, 8]])

b = np.array([1, 0, 0])

def solveLGS(A, b):
    return np.linalg.solve(A, b)

if __name__ == "__main__":
    y = solveLGS(L, b)
    x1 = solveLGS(R, y)
    print(x1)
    
    x2 = solveLGS(A, b)
    print(x2)


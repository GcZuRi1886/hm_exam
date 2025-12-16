import numpy as np

def max_relative_error(A, b, A_err, b_err, A_cond):
    A_norm = np.linalg.norm(A, np.inf)
    A_err_norm = np.linalg.norm(A - A_err, np.inf)
    b_norm = np.linalg.norm(b, np.inf)
    b_err_norm = np.linalg.norm(b - b_err, np.inf)

    left_mult = A_cond / (1 - A_cond * A_err_norm / A_norm)
    right_mult = (A_err_norm / A_norm) + (b_err_norm / b_norm)

    return left_mult * right_mult
    
def obs_relative_error(x, x_err):
    x_norm = np.linalg.norm(x, np.inf)
    x_err_norm = np.linalg.norm(x - x_err, np.inf)

    return x_err_norm / x_norm


def error_analysis(A, b, A_err, b_err):
    A = np.array(A)
    b = np.array(b)
    A_err = np.array(A_err)
    b_err = np.array(b_err)

    try:
        x = np.linalg.solve(A, b)
        x_err = np.linalg.solve(A_err, b_err)
    except np.linalg.LinAlgError:
        return None, None, None, None

    A_cond = np.linalg.cond(A, p=np.inf)

    dx_max = max_relative_error(A, b, A_err, b_err, A_cond)
    dx_obs = obs_relative_error(x, x_err)

    return x, x_err, dx_max, dx_obs

if __name__ == "__main__":

    A = [[1/2, -1/2], [1/6, 1/3]]
    b = [1, -1/6]
    A_err = [[1/2, -1/2], [0.2, 0.3]]
    b_err = [1, -0.2]

    result = error_analysis(A, b, A_err, b_err)

    if result is not None:
        x, x_err, dx_max, dx_obs = result
        print("Solution x:", x)
        print("Perturbed Solution x_err:", x_err)
        print("Maximum Relative Error dx_max:", dx_max)
        print("Observed Relative Error dx_obs:", dx_obs)


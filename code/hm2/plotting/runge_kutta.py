import numpy as np
import matplotlib.pyplot as plt

def plot_runge_kutta_4(num_result: np.ndarray, exact_solution: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.plot(num_result[:, 0], num_result[:, 1], label='Runge-Kutta 4', marker='o')
    plt.plot(exact_solution[:, 0], exact_solution[:, 1], label='Exact Solution', linestyle='--')
    
    plt.title('Runge-Kutta 4 Method')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid()
    plt.show()


def plot_general_runge_kutta(results: list[tuple[np.ndarray, str]]):
    plt.figure(figsize=(10, 6))
    for num_result, label in results:
        plt.plot(num_result[:, 0], num_result[:, 1], label=label, marker='o')
    
    plt.title('General Runge-Kutta Method')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid()
    plt.show()

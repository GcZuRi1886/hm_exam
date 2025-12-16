import FabianOdermatt_S09_Aufg2 as aufg2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

results = []

for i in range(1, 1000):
    A = np.random.rand(100, 100)
    b = np.random.rand(100, 1)

    A_err = A + np.random.rand(100, 100) * 10**-5
    b_err = b + np.random.rand(100, 1) * 10**-5

    result = aufg2.error_analysis(A, b, A_err, b_err)
    if result is not None:
        x, x_err, dx_max, dx_obs = result
        dx_ratio = dx_max / dx_obs
        results.append((dx_max, dx_obs, dx_ratio))

def plot_results(results):
    dx_max_values = [res[0] for res in results]
    dx_obs_values = [res[1] for res in results]
    dx_ratio_values = [res[2] for res in results]

    plt.figure(figsize=(12, 6))

    plt.semilogy(dx_max_values, label='Maximaler Fehler dx_max', color='blue', marker='o', linestyle='None')
    plt.semilogy(dx_obs_values, label='Beobachteter Fehler dx_obs', color='orange', marker='o', linestyle='None')
    plt.semilogy(dx_ratio_values, label='Verhältnis dx_max/dx_obs', color='green', marker='o', linestyle='None')
    plt.xlabel('Testfall Index')
    plt.ylabel('Fehlerwerte (logarithmisch)')
    plt.title('Fehleranalyse der Lösung von linearen Gleichungssystemen')
    plt.legend()
    plt.show()

plot_results(results)

# Dx_max ist in der Regel grösser als Dx_obs, was darauf hinweist, dass die theoretischen Abschätzungen realistisch sind

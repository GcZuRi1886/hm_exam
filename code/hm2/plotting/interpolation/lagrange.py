import matplotlib.pyplot as plt

def plot_interpolation(x, y, x_fit, y_fit):
    plt.xlabel('Jahr')
    plt.ylabel('Haushalte mit Computer (%)')
    plt.title('Anteil der Haushalte mit Computer über die Jahre')
    plt.plot(x, y, 'ro', label='Datenpunkte')
    plt.plot(x_fit, y_fit, 'b-', label='Lagrange-Interpolation')
    plt.xlim(1975, 2020)
    plt.ylim(-100, 250)
    plt.legend()
    plt.grid()
    plt.show()

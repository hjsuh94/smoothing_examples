import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

grid_size = 50
x_grid = np.linspace(0, 1, grid_size)
y_grid = np.linspace(0, 1, grid_size)
results = np.zeros((grid_size, grid_size))
results_smooth = np.zeros((grid_size, grid_size))

def compute_cost(x, y):

    u1_x = x
    u2_x = y

    if u1_x <= 0.5:
        u1_x = u1_x
        u1_y = 1.0

        if (u2_x <= 0.333):
            u2_x = u2_x
            u2_y = 2.0
        else:
            u2_x = 2.0
            u2_y = 0.0
    else:
        u1_x = 0.0
        u1_y = 0.0
        u2_x = 0.0
        u2_y = 0.0

    cost = (u1_x ** 2.0 + u2_x ** 2.0) + 0.2 * (u1_y ** 2.0 + u2_y ** 2.0)
    return -cost

sample_size = 1000

for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        results[i,j] = compute_cost(x_grid[i], y_grid[j])
        samples = np.random.normal(0, 0.1, (sample_size,2))
        for k in range(sample_size):
            results_smooth[i,j] += compute_cost(x_grid[i] + samples[k,0], y_grid[j] + samples[k,1])
        results_smooth[i,j] /= sample_size

X, Y = np.meshgrid(y_grid, x_grid)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, results, cmap=cm.viridis, alpha=0.6)
surf = ax.plot_surface(X, Y, results_smooth, cmap=cm.plasma, alpha=0.8)
plt.xlabel('u2')
plt.ylabel('u1')
plt.show()

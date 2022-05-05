import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

def map_to_billiard(x, y, xmax, ymax):
    x_reflect = np.remainder(int(x / xmax), 2)
    y_reflect = np.remainder(int(y / ymax), 2)

    x_pos = np.remainder(x, xmax)
    y_pos = np.remainder(y, ymax)

    if (x_reflect):
        x_pos = xmax - x_pos
    if (y_reflect):
        y_pos = ymax - y_pos

    return np.array([x_pos, y_pos])

"""
# TEST CODE FOR map_to_billiard
test_ray = np.zeros((1000,2))
test_ray[:,0] = np.linspace(0,6,1000) * np.random.rand()
test_ray[:,1] = np.linspace(0,6,1000) * np.random.rand()
billiard_map = np.zeros((1000, 2))

for i in range(1000):
    billiard_map[i,:] = map_to_billiard(test_ray[i,0], test_ray[i,1], 1, 2)

plt.figure()
plt.plot(test_ray[:,0], test_ray[:,1], 'k-')
plt.plot(billiard_map[:,0], billiard_map[:,1], 'g-')
plt.axis('equal')

rect = patches.Rectangle((0, 0), 1, 2, linewidth=1, edgecolor='r', facecolor='none')
plt.gca().add_patch(rect)

plt.show()
"""

grid_size = 100
v_grid = np.linspace(0, 4, grid_size)
theta_grid = np.linspace(0, np.pi/2, grid_size)
results = np.zeros((grid_size, grid_size))

for i in range(len(v_grid)):
    for j in range(len(theta_grid)):
        v = v_grid[i]
        theta = theta_grid[j]

        pf = map_to_billiard(v * np.cos(theta), v * np.sin(theta), 1, 2)
        pg = np.array([0.5, 1.0])
        results[i,j] = np.linalg.norm(pf - pg, 2)
        results[i,j] += 0.3 * v

X, Y = np.meshgrid(theta_grid, v_grid)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, results, cmap=cm.viridis)
#surf = ax.plot_surface(X, Y, results_smooth, cmap=cm.Reds)
plt.xlabel('theta')
plt.ylabel('V')
plt.show()        

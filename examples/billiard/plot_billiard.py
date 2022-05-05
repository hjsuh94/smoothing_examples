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

# TEST CODE FOR map_to_billiard
v = 3.9
theta = 0.88
test_ray = np.zeros((1000,2))
test_ray[:,0] = np.linspace(0,1,1000) * v * np.cos(theta)
test_ray[:,1] = np.linspace(0,1,1000) * v * np.sin(theta)
billiard_map = np.zeros((1000, 2))

for i in range(1000):
    billiard_map[i,:] = map_to_billiard(test_ray[i,0], test_ray[i,1], 1, 2)

plt.figure(figsize=(4,8))
plt.plot(billiard_map[:,0], billiard_map[:,1], color='springgreen')
plt.plot(billiard_map[-1,0], billiard_map[-1,1], 'ro')
plt.plot(0.5, 1.0, 'bo')
rect = patches.Rectangle((0, 0), 1, 2, linewidth=1, edgecolor='k', facecolor='none')
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 2.5])
plt.gca().add_patch(rect)
plt.axis('equal')
plt.axis('off')

plt.show()

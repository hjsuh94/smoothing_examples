import gym

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

grid_size = 100
ufx_grid = np.linspace(-0.4, 0.4, grid_size)
ufy_grid = np.linspace(-0.4, 0.4, grid_size)
results = np.zeros((grid_size, grid_size))

env = gym.make("Carrot-v0")

for i in tqdm(range(len(ufx_grid))):
    for j in range(len(ufy_grid)):
        np.random.seed(0)
        ob = env.reset()
        obs, reward, done, _ = env.step([-0.4, 0.4, ufx_grid[i], ufy_grid[j]])
        results[i,j] = reward

X, Y = np.meshgrid(ufx_grid, ufy_grid)

dir = "examples/carrots/data/"

np.save(os.path.join(dir, "X.npy"), X)
np.save(os.path.join(dir, "Y.npy"), Y)
np.save(os.path.join(dir, "results.npy"), results)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, results, cmap=cm.viridis)
plt.xlabel('ufx')
plt.ylabel('ufy')
plt.show()

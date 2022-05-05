import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 

import os

dir = "examples/payload_double_integrator/data"

X = np.load(os.path.join(dir, "X.npy"))
Y = np.load(os.path.join(dir, "Y.npy"))
results = np.load(os.path.join(dir, "results.npy"))
results_smooth = np.load(os.path.join(dir, "results_smooth.npy"))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, results, cmap=cm.viridis, alpha=0.6)
surf = ax.plot_surface(X, Y, results_smooth, cmap=cm.plasma, alpha=0.8)
plt.xlabel('Switching Time')
plt.ylabel('Acceleration')
plt.show()
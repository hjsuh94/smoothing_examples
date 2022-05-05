import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 

import os

dir = "examples/carrots/data"

X = np.load(os.path.join(dir, "X.npy"))
Y = np.load(os.path.join(dir, "Y.npy"))
results = np.load(os.path.join(dir, "results.npy"))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, results, cmap=cm.viridis, alpha=1.0)
plt.show()
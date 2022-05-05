import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm

def momentum_transfer(theta):
    if np.abs(theta) > np.pi/4:
        return 0
    else:
        return np.sin(theta)

sample_size = 1000
theta = np.linspace(-np.pi/2, np.pi/2, 1000)
momentums = np.zeros(1000)
momentums_smooth = np.zeros(1000)
for i in range(len(theta)):
    momentums[i] = momentum_transfer(theta[i])
    momentums_smooth[i] = 0.0
    samples = np.random.normal(0, 0.1, sample_size)
    for k in range(sample_size):
        momentums_smooth[i] += momentum_transfer(theta[i] + samples[k])
    momentums_smooth[i] /= sample_size

plt.figure()
plt.plot(theta, -momentums, 'k-', label='original cost')
plt.plot(theta, -momentums_smooth, 'r-', label='randomized smoothing')
plt.xlabel('theta')
plt.ylabel('transferred momentum')
plt.legend()
plt.show()
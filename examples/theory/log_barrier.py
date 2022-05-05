import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import erf

def gaussian(x):
    mu = 0
    sig = 1
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


xhalf = np.linspace(1e-22,1,1000)
x = np.linspace(-1,1,1000)

stretch = 4.0

colormap = cm.get_cmap("Blues")

plt.figure()
for i in range(1,6):
    jm = colormap(i / 6)
    plt.plot(x, 5 * i * (erf(-3 * i * x) + 1), color=jm, label="erf_" + str(i))
    plt.plot(x, 10 * i * (-3 * i * x > 0), '--', color=jm)
plt.plot(xhalf, -np.log(xhalf), 'r-', label="log barrier")    
plt.xlabel('x: decision variable')
plt.ylabel('cost')

plt.legend()
plt.show()

xhalf = np.linspace(0.05,1,1000)

plt.figure()
for i in range(1,6):
    jm = colormap(i / 6)
    plt.plot(x, 5 * i * (gaussian(-3 * i * x)), color=jm, label="N_" + str(i))
plt.plot(xhalf, 0.5 * 1./xhalf, 'r-', label="1/x")    
plt.xlabel('x: decision variable')
plt.ylabel('cost gradient')

plt.legend()
plt.show()



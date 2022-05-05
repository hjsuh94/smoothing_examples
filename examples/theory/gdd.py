import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import erf

x = np.linspace(0,1,1000)

y = np.zeros(1000)
y_smooth = np.zeros(1000)

y_half_left = np.zeros(1000)
y_half_left_smooth = np.zeros(1000)

y_half_right = np.zeros(1000)
y_half_right_smooth = np.zeros(1000)

def original(x):
    if x < 0.5:
        y = 2.0 * (x - 0.5) ** 2.0
    else:
        y = 1.0
    return y

def half_left(x):
    if x < 0.5:
        y = 2.0 * (x - 0.5) ** 2.0
    else:
        y = 0.0
    return y

def half_right(x):
    if x < 0.5:
        y = 0.0
    else:
        y = 1.0             
    return y   

for i in range(1000):
    y[i] = original(x[i])
    y_half_left[i] = half_left(x[i])
    y_half_right[i] = half_right(x[i])

    samples = np.random.normal(0, 0.1, 10000)

    for k in range(len(samples)):
        y_smooth[i] += original(x[i] + samples[k])
        y_half_left_smooth[i] += half_left(x[i] + samples[k])
        y_half_right_smooth[i] += half_right(x[i] + samples[k])

    y_smooth[i] /= len(samples)
    y_half_left_smooth[i] /= len(samples)    
    y_half_right_smooth[i] /= len(samples)        
        
plt.figure()
plt.plot(x,y, 'k-')
plt.axis('off')
plt.show()

plt.figure()
plt.plot(x,y_half_left, color='magenta')
plt.axis('off')
plt.show()

plt.figure()
plt.plot(x,y_half_right, color='springgreen')
plt.axis('off')
plt.show()

plt.figure()
plt.plot(x,y_smooth, 'k-')
plt.axis('off')
plt.show()

plt.figure()
plt.plot(x,y_half_left_smooth, color='magenta')
plt.axis('off')
plt.show()

plt.figure()
plt.plot(x,y_half_right_smooth, color='springgreen')
plt.axis('off')
plt.show()

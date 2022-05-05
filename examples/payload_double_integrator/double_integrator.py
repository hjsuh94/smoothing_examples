import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import os

def controller(tc, params):
    mag = params[0]
    switch_time = params[1]

    if tc < switch_time:
        return mag 
    else:
        return -mag

def simulate_system(x, u, h):
    # x = [p, v]
    # Integrate symplectic.
    pc = x[0]
    vc = x[1]

    vn = vc + h * u
    pn = pc + h * vn

    return [pn, vn]

def rollout(x0, h, T, params):
    x_trj = np.zeros((T+1,2))
    u_trj = np.zeros(T)
    x_trj[0,:] = x0
    tc = 0
    for t in range(T):
        u_trj[t] = controller(tc, params)
        x_trj[t+1,:] = simulate_system(x_trj[t,:], u_trj[t], h)
        tc += h

    return x_trj, u_trj

def evaluate_cost(x_trj, u_trj, params):

    cost = 0.0
    fallen = False


    Q = np.diag([100.0, 0.1])
    x_final = x_trj[-1,:]
    cost += x_final @ Q @ x_final

    cost += 10 * params[1]

    for t in range(len(u_trj)):
        if u_trj[t] > 0.9:
            fallen = True
            
    if (fallen): 
        cost = 150.0

    return cost 
        
x0 = [-0.5, 0]
h = 0.01 
T = 1000

grid_size = 50
mag = np.linspace(0.0001,1.8, grid_size)
switch_time = np.linspace(0.0001, T * h / 2, grid_size)
results = np.zeros((grid_size, grid_size))
results_smooth = np.zeros((grid_size, grid_size))

sample_size = 1000
 
for i in tqdm(range(len(mag))):
    for j in range(len(switch_time)):
        params = [mag[i], switch_time[j]]
        x_trj, u_trj = rollout(x0, 0.01, 100, params)
        results[i,j] = evaluate_cost(x_trj, u_trj, params)

        samples = np.random.normal([0, 0], [0.1,0.3], (sample_size,2))
        for k in range(sample_size):
            params = [mag[i] + samples[k,0], switch_time[j] + samples[k,1]]
            x_trj, u_trj = rollout(x0, 0.01, 100, params)
            results_smooth[i,j] += evaluate_cost(x_trj, u_trj, params)
        results_smooth[i,j] /= sample_size

X, Y = np.meshgrid(switch_time, mag)

dir = "examples/payload_double_integrator/data/"

np.save(os.path.join(dir, "X.npy"), X)
np.save(os.path.join(dir, "Y.npy"), Y)
np.save(os.path.join(dir, "results.npy"), results)
np.save(os.path.join(dir, "results_smooth.npy"), results_smooth)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, results, cmap=cm.Purples)
surf = ax.plot_surface(X, Y, results_smooth, cmap=cm.Reds)
plt.xlabel('Switching Time')
plt.ylabel('Acceleration')
plt.show()
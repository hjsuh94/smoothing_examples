import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import os

def controller(tc, params):
    u = params
    if tc == 0:
        return params
    else:
        return 0

def simulate_system(x, u, h):
    # x = [p, v]
    # Integrate symplectic.
    pc = x[0]
    vc = x[1]

    vn = vc + h * (u + 10.0 * np.sin(pc))
    pn = pc + h * vn

    if pn >= 0.6:
        pn = 0.6
        vn = -0.9 * vn

    if pn <= -0.6:
        pn = -0.6
        vn = -0.9 * vn

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

def evaluate_cost(x_trj, u_trj):

    cost = 0.0
    Q = np.diag([10.0, 10.0])
    R = 0.001

    x_final = x_trj[-1,:]
    u_init = u_trj[0]
    cost += x_final @ Q @ x_final
    cost += u_init ** 2.0 * R 
    return cost 
        
x0 = [0.2, 0.6]
h = 0.01 
T = 100

grid_size = 100
v0 = np.linspace(0, 2.0, grid_size)
u = np.linspace(-100.0, 100.0, grid_size)
results = np.zeros((grid_size, grid_size))
results_smooth = np.zeros((grid_size, grid_size))

sample_size = 1000

for i in tqdm(range(len(v0))):
    for j in range(len(u)):
        x0 = [0.2, v0[i]]
        x_trj, u_trj = rollout(x0, h, T, u[j])
        results[i,j] = evaluate_cost(x_trj, u_trj)

        """
        samples = np.random.normal([0, 0], [0.1,0.3], (sample_size,2))
        for k in range(sample_size):
            params = [mag[i] + samples[k,0], switch_time[j] + samples[k,1]]
            x_trj, u_trj = rollout(x0, 0.01, 100, params)
            results_smooth[i,j] += evaluate_cost(x_trj, u_trj, params)
        results_smooth[i,j] /= sample_size
        """

X, Y = np.meshgrid(u, v0)


dir = "examples/push_recovery/data/"

np.save(os.path.join(dir, "X.npy"), X)
np.save(os.path.join(dir, "Y.npy"), Y)
np.save(os.path.join(dir, "results.npy"), results)
#np.save(os.path.join(dir, "results_smooth.npy"), results_smooth)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, results, cmap=cm.viridis)
#surf = ax.plot_surface(X, Y, results_smooth, cmap=cm.Reds)
plt.xlabel('Impulse')
plt.ylabel('Velocity')
plt.show()

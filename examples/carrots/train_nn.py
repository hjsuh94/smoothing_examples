import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from tqdm import tqdm

import os
import torch.nn as nn
import torch.optim as optim
import torch 
import time

dir = "examples/carrots/data"

X = np.load(os.path.join(dir, "X.npy"))
Y = np.load(os.path.join(dir, "Y.npy"))
results = np.load(os.path.join(dir, "results.npy"))

print(results.shape)

data = np.zeros((10000,3))
count = 0
for i in range(100):
    for j in range(100):
        data[count,0] = X[i,j]
        data[count,1] = Y[i,j]
        data[count,2] = results[i,j]
        count += 1

class DynamicsNLP(nn.Module):
    def __init__(self):
        super(DynamicsNLP, self).__init__()

        self.dynamics_mlp = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.dynamics_mlp(x)

dynamics_net = DynamicsNLP()
dynamics_net.train()
optimizer = optim.Adam(dynamics_net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500)
criterion = nn.MSELoss()

num_iter = 1000
for iter in tqdm(range(num_iter)):
    optimizer.zero_grad()
    output = dynamics_net(torch.Tensor(data[:,0:2]))
    loss = criterion(output, torch.Tensor(data[:,2]).unsqueeze(1))
    loss.backward()
    optimizer.step()
    scheduler.step()

dynamics_net.eval()

results_smooth = np.zeros((100, 100))
count = 0
for i in range(100):
    for j in range(100):
        input = torch.Tensor(np.array([X[i,j], Y[i,j]]))
        results_smooth[i,j] = dynamics_net(input).detach().numpy()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, results, cmap=cm.viridis, alpha=0.6)
surf = ax.plot_surface(X, Y, results_smooth, cmap=cm.plasma, alpha=0.8)
plt.show()



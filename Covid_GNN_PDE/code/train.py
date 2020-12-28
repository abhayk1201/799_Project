import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt

from pdegs.dynamics import DynamicsFunction
from pdegs.integrators import ODEAdjointIntegrator
from pdegs.models import MPNNDiff, Model3

import utils


# Can be replaced by argparse
Config = namedtuple(
    "Config", 
    [
        "d", "hs_1", "hs_2", "method", "rtol", 
        "atol", "device", "batch_size", "lr", 
        "epochs", "model_path", "data_path",
        "tb_log_dir",
    ]
)

args = Config(
    d=40,
    hs_1=60,
    hs_2=0,
    method="euler",  #euler
    rtol=1.0e-7,
    atol=1.0e-7,
    device="cpu", #CHANGE cuda
    batch_size=None,  # Use None for full batch
    lr=0.000001,
    epochs=10000,
    model_path="./model_covid_state_daily_norm_10k.pth",
    data_path="../data/covid_state_daily_norm_train/",
    tb_log_dir="./tmp_logs_covid/",
)

n_s = 1

print("NUMBER OF RANDOM SIMULATIONS :", n_s)

print(args)
device = torch.device(args.device)
writer = SummaryWriter(log_dir=args.tb_log_dir)

# Create model
msg_net = nn.Sequential(
    nn.Linear(4, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.d)
)
aggr_net = nn.Sequential(
    nn.Linear(args.d+1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, 1)
)
model = MPNNDiff(msg_net, aggr_net)

model.apply(utils.weights_init)
print("Num. of params: {:d}".format(utils.get_parameters_count(model)))

F = DynamicsFunction(model).to(device)

# Create integrator
adj_integr = ODEAdjointIntegrator()

# Prepare data
data = utils.read_pickle(['t', 'x', 'u'], args.data_path)
dataset = utils.generate_torchgeom_dataset(data, sig=0.0)

# #########
sim_inds = [0] #np.random.choice(len(dataset), n_s, replace=False)
print(f'sim_inds = {sim_inds}')
dataset = [ds for i, ds in enumerate(dataset) if i in sim_inds]
print(f'dataset length: {len(dataset)}')
# #########

if args.batch_size is None:
    batch_size = len(dataset)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer, loss
optimizer = optim.Rprop(F.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

# Training
ts = dataset[0].t.shape[0]  # assumes the same time grid for all sim-s.
print("%%%%%%%%%%%%%%%%%%")
print(dataset[0])
for epoch in range(args.epochs):
    losses = torch.zeros(len(loader))
    
    for i, dp in enumerate(loader):
        optimizer.zero_grad()

        params_dict = {"edge_index": dp.edge_index.to(device), "pos": dp.pos.to(device)}
        F.update_params(params_dict)

        y0 = dp.x.to(device)
        t = dp.t[0:ts].to(device)
        y_pd = adj_integr.integrate(
            F, y0, t, method=args.method, rtol=args.rtol, atol=args.atol)
        
        sims = []
        for j in range(batch_size):
            sims.append(dp.y[j*ts:(j+1)*ts])  # TODO: just transpose 0, 1 ?
        y_gt = torch.cat(sims, dim=1)

        loss = loss_fn(y_pd, y_gt.to(device))
        loss.backward()
        optimizer.step()

        losses[i] = loss.item()
        
    writer.add_scalar("train_loss/"+str(args), losses.mean(), epoch)
    
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print("epoch {:>5d} | train loss: {:>7.12f}".format(epoch, losses.mean()))
        torch.save(F.state_dict(), args.model_path)
    if losses.mean() <= 0.0001:
        break

dp = dataset[0]
params_dict = {"edge_index": dp.edge_index.to(device), "pos": dp.pos.to(device)}
F.update_params(params_dict)
y0 = dp.x.to(device)
t = dp.t[0:ts].to(device)
y_pd = adj_integr.integrate(
    F, y0, t, method=args.method, rtol=args.rtol, atol=args.atol)

print("Plotting...")
print(y_pd.cpu().detach().numpy().shape)
print(dp.y.numpy().shape)
print(dataset[0].t.shape)
print(dataset[0].pos.shape)
plt.figure(0)
utils.plot_fields(
    t=dataset[0].t,
    coords=dataset[0].pos,
    fields={
        "y_pd": y_pd.cpu().detach().numpy(),
        "y_gt": dp.y.numpy(),
    },
    save_path="./tmp_covid_state_daily_norm_10k/",
    delay=0.0001,
)
plt.show()

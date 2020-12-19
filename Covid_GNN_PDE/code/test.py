import torch
import torch.nn as nn

from torch_geometric.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt

from pdegs.dynamics import DynamicsFunction
from pdegs.integrators import ODEAdjointIntegrator
from pdegs.models import Model3, MPNNDiff

from collections import namedtuple

import utils


# can be replaced by argparse
Config = namedtuple(
    "Config", 
    [
        "d", "hs_1", "hs_2", "method", "rtol", 
        "atol", "device", "model_path", "data_path"
    ]
)

args = Config(
    d=40,
    hs_1=60,
    hs_2=0,
    method="adams",
    rtol=1.0e-7,
    atol=1.0e-7,
    device="cuda",
    model_path="./model_gnode_n3000_run_4.pth",
    data_path="../data/subs_paper/convdiff_2pi_n3000_t21_test/",
)

device = torch.device(args.device)

# Create model
# Model3 
# L1_msg_net = nn.Sequential(nn.Linear(2, args.hs_1), nn.Tanh(), nn.Linear(args.hs_1, args.d))
# L1_aggr_net = nn.Sequential(nn.Linear(args.d, args.hs_1), nn.Tanh(), nn.Linear(args.hs_1, args.d))
# L2_msg_net = nn.Sequential(nn.Linear(4+args.d, args.hs_2), nn.Tanh(), nn.Linear(args.hs_2, args.d))
# L2_aggr_net = nn.Sequential(nn.Linear(args.d, args.hs_2), nn.Tanh(), nn.Linear(args.hs_2, 1))
# model = Model3(L1_msg_net, L1_aggr_net, L2_msg_net, L2_aggr_net)

# MPNN
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

F = DynamicsFunction(model).to(device)
F.load_state_dict(torch.load(args.model_path, map_location=device))

# Create integrator
adj_integr = ODEAdjointIntegrator()

# Prepare data
data = utils.read_pickle(['t', 'x', 'u'], args.data_path)
print(data['t'].shape)

dataset = utils.generate_torchgeom_dataset(data)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Loss
loss_fn = nn.MSELoss()

# Testing
diffs_over_time = []
losses = torch.zeros(len(loader))

inds_of_sims_to_show = set([])

for i, dp in enumerate(loader):
    params_dict = {"edge_index": dp.edge_index.to(device), "pos": dp.pos.to(device)}
    F.update_params(params_dict)

    y0 = dp.x.to(device)
    t = dp.t.to(device)
    y_pd = adj_integr.integrate(
        F, y0, t, method=args.method, rtol=args.rtol, atol=args.atol)
    
    y_gt = dp.y.to(device)

    loss = loss_fn(y_pd, y_gt)

    losses[i] = loss.item()

    u = y_gt.cpu().detach().numpy()
    u_pd = y_pd.cpu().detach().numpy()
    u_mean = u.mean(axis=1).reshape(-1)

    # diffs = [((u[i] - u_pd[i])**2).mean() for i in range(len(u))]
    
    eps = 1.0e-6
    diffs = [np.linalg.norm(u[i].reshape(-1) - u_pd[i].reshape(-1)) / (np.linalg.norm(u[i].reshape(-1)) + eps) for i in range(len(u))]
    
    diffs_over_time.append(diffs)

    print("test case {:>5d} | test loss: {:>7.12f}".format(i, losses[i]))

    if i in inds_of_sims_to_show:
        print("Plotting...")
        utils.plot_grid(dataset[i].pos.cpu().detach().numpy())
        plt.figure(0)
        utils.plot_fields(
            t=dataset[i].t,
            coords=dataset[i].pos,
            fields={
                "y_pd": y_pd.cpu().detach().numpy(),
                "y_gt": dp.y.numpy(),
            },
            save_path="./tmp_figs/",
            delay=0.0001,
        )
        plt.show()

    # if i == 2:  # 3 for grids, 2 for time points
    #     break

# comp_dict = np.load("./paper_exps/comp_dict_paper_diff_step_size.npy", allow_pickle='TRUE').item()
comp_dict = {}

comp_dict_upd = {
    # 'coords_true': dataset[2].pos.numpy(), 
    # 'fields_true': dataset[2].y.numpy(),
    'coords_t_11': dataset[2].pos.numpy(),
    'fields_t_11': y_pd.cpu().detach().numpy(),
    
}
comp_dict.update(comp_dict_upd)
np.save("./paper_exps/comp_dict_paper_diff_step_size.npy", comp_dict)
print(comp_dict.keys())

print("Plotting diffs...")
plt.figure(0)
t = dataset[0].t.numpy()

for diff in diffs_over_time:
    plt.semilogy(t, diff, alpha=0.5)
plt.vlines(0.2, 0, 0.00002, color='gray')

plt.ylabel("MSE")
plt.xlabel("t (sec)")

plt.savefig("diffs.png")

diffs_over_time = np.array(diffs_over_time)
print("diffs_over_time.shape", diffs_over_time.shape)
print("diffs_over_time.mean", diffs_over_time.mean())

# save_to = "./plot_data/time_11tpts.npy"
# np.save(save_to, diffs_over_time)
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('seaborn')


class EdgeConvBase(MessagePassing):
    # Adapted https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv
    def __init__(
        self, msg_net, aggr_net, aggr='mean', 
        neighbor_loc="position", self_val=True, **kwargs
    ):
        super(EdgeConvBase, self).__init__(aggr=aggr, flow="target_to_source", **kwargs)
        self.msg_net = msg_net
        self.aggr_net = aggr_net
        self.neighbor_loc = neighbor_loc
        self.self_val = self_val
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, pos):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        if self.neighbor_loc == "distance":
            inputs = torch.cat([x_i, x_j-x_i, (pos_j-pos_i).norm(dim=1).view(-1, 1)], dim=1)
        elif self.neighbor_loc == "position":
            inputs = torch.cat([x_i, x_j-x_i, pos_j-pos_i], dim=1)
            # inputs = torch.cat([x_i, x_j-x_i], dim=1)
        elif self.neighbor_loc == "radial":
            dist = (pos_j-pos_i).norm(dim=1).view(-1, 1)
            pos_vec = pos_j - pos_i
            angle = torch.atan2(pos_vec[:, 1], pos_vec[:, 0]).view(-1, 1)
            inputs = torch.cat([x_i, x_j-x_i, dist, angle], dim=1)
        return self.msg_net(inputs)

    def update(self, aggr_out, x):
        if self.self_val:
            inp = torch.cat((x, aggr_out), dim=1)
        else:
            inp = aggr_out
        return self.aggr_net(inp)

    def __repr__(self):
        return '{}(msg_nn={}, aggr_nn={})'.format(
            self.__class__.__name__, self.msg_net, self.aggr_net)


class MyEdgeConv(EdgeConvBase):
    def message(self, x_i, x_j, pos_i, pos_j):
        inputs = torch.cat([x_i, x_j], dim=1)
        return self.msg_net(inputs)


class MPNNDiff(nn.Module):
    def __init__(self, msg_net, aggr_net):
        super(MPNNDiff, self).__init__()
        self.L1 = EdgeConvBase(
            msg_net, aggr_net, aggr='mean', neighbor_loc="position", self_val=True)

    def forward(self, x, edge_index, pos):
        return self.L1(x, edge_index.long(), pos)


class MPNNDiffK2(nn.Module):
    def __init__(self, msg_net_1, aggr_net_1, msg_net_2, aggr_net_2):
        super(MPNNDiffK2, self).__init__()
        self.L1 = EdgeConvBase(
            msg_net_1, aggr_net_1, aggr='mean', neighbor_loc="position", self_val=True)
        self.L2 = EdgeConvBase(
            msg_net_2, aggr_net_2, aggr='mean', neighbor_loc="position", self_val=True)

    def forward(self, x, edge_index, pos):
        L1_out = self.L1(x, edge_index.long(), pos)
        L2_out = self.L2(L1_out, edge_index.long(), pos)
        return L2_out
import torch.nn as nn
import torch.nn.functional as F

from module.model import *


class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        from module.model.layer.masked_gcn_layer import MaskedGCNConv, MaskedLinear
        self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
        self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
        self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True:
            return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x

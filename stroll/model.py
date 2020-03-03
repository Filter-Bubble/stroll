import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

import numpy as np
from sklearn.metrics import confusion_matrix

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        feats = self.linear(node.data['h'])
        if self.activation is not None:
            feats = self.activation(feats)
        return {'h' : feats}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):

        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.sum(msg='m', out='h')

        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self, in_feats=16, h_dims=16, out_feats_a=2, out_feats_b=16):
        super(Net, self).__init__()
        self.gcn1 = GCN(in_feats, h_dims, F.relu)
        self.gcn2 = GCN(h_dims, h_dims, F.relu)
        self.linear1a = nn.Linear(h_dims, out_feats_a)
        self.linear1b = nn.Linear(h_dims, out_feats_b)

    def forward(self, g):
        x = self.gcn1(g, g.ndata['v'])
        x = self.gcn2(g, x)
        xa = self.linear1a(x)
        xb = self.linear1b(x)
        return xa, xb

    def evaluate(self, g):
        self.eval()
        with torch.no_grad():
            logits_a, logits_b = self(g)

            _, pred_a = torch.max(logits_a, dim=1)
            _, pred_b = torch.max(logits_b, dim=1)

            targets_a = g.ndata['frame']
            targets_b = g.ndata['role']

            correct_a = torch.sum(pred_a == targets_a)
            correct_b = torch.sum(pred_b == targets_b)

            acc_a = correct_a.item() * 1.0 / len(targets_a)
            acc_b = correct_b.item() * 1.0 / len(targets_b)

            conf_a = confusion_matrix(pred_a, targets_a, labels=np.arange(2))
            conf_b = confusion_matrix(pred_b, targets_b, labels=np.arange(21))

            return  acc_a, acc_b, conf_a, conf_b

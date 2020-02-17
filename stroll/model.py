import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

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
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(768, 16, F.relu)
        self.gcn2 = GCN(16, 21, None)

    def forward(self, g):
        x = self.gcn1(g, g.ndata['v'])
        x = self.gcn2(g, x)
        return x

    def evaluate_role(self, g):
        self.eval()
        with torch.no_grad():
            logits = self(g)
            _, indices = torch.max(logits, dim=1)
            targets = g.ndata['role']
            correct = torch.sum(indices == targets)
            return correct.item() * 1.0 / len(targets)

    def evaluate_frame(self, g):
        self.eval()
        with torch.no_grad():
            logits = self(g)
            _, indices = torch.max(logits, dim=1)
            targets = g.ndata['frame']
            correct = torch.sum(indices == targets)
            return correct.item() * 1.0 / len(targets)


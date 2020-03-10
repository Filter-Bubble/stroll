import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

from .labels import role_codec, frame_codec

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
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.apply_mod = NodeApplyModule(self.in_feats, self.out_feats, self.activation)

    def forward(self, g, feature):

        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.mean(msg='m', out='h')

        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


# https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/4_rgcn.html
# simplify by setting num_bases = num_rels = 3
class RGCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation, skip=False):
        super(RGCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.skip = skip

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(3, self.in_feats, self.out_feats))
        # nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, g, feature):
        weight = self.weight

        # At each edge, multiply the state h from the source node
        # with a linear weight W_(edge_type)
        def rgcn_msg(edges):
            # n.shape 2339
            # weight[edges.data['rel_type']].shape 2339,85,64
            # edges.src['h'].unsqueeze(1).shape) 2339,1,85
            w = weight[edges.data['rel_type']]
            n = edges.data['norm']
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = torch.bmm(n.reshape(-1,1,1), msg.unsqueeze(1)).squeeze()

            return {'m': msg}

        # At each node, we want the summed messages W_(edge_type) \dot h
        # from the incomming edges

        # Add a skip connection around the (\sum W h)
        if self.skip:
            rgcn_reduce = fn.sum(msg='m', out='Swh')
        else:
            rgcn_reduce = fn.sum(msg='m', out='h')

        # Apply activation to the sum(in_edges) W_(edge_type) \dot h
        # TODO: add bias?
        def rgcn_apply(nodes):
            h = nodes.data['h']
            if self.skip:
                Swh = nodes.data['Swh']
                h = self.activation(h + Swh)
            else:
                h = self.activation(h)
            return {'h': h}

        g.ndata['h'] = feature
        g.update_all(rgcn_msg, rgcn_reduce, rgcn_apply)
        return g.ndata.pop('h')

class Net(nn.Module):
    def __init__(self, in_feats=16, h_dims=16, out_feats_a=2, out_feats_b=16):
        super(Net, self).__init__()
        self.rgcn1 = RGCN(in_feats, h_dims, F.relu, skip=False)
        self.rgcn2 = RGCN(h_dims, h_dims, F.relu, skip=True)
        self.rgcn3 = RGCN(h_dims, h_dims, F.relu, skip=True)
        self.linear1a = nn.Linear(h_dims, out_feats_a)
        self.linear1b = nn.Linear(h_dims, out_feats_b)

        self.loss_weight1 = torch.nn.Parameter(torch.tensor([0.25]))
        self.loss_weight2 = torch.nn.Parameter(torch.tensor([3.]))

        nn.init.xavier_uniform_(self.linear1a.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.linear1b.weight, nn.init.calculate_gain('sigmoid'))

    def forward(self, g):
        #    RGCN1 -> RGCN2 -> RGCN3 -> Linear
        x = self.rgcn1(g, g.ndata['v'])
        x = self.rgcn2(g, x)
        x = self.rgcn3(g, x)
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

            acc_a = balanced_accuracy_score(targets_a, pred_a)
            acc_b = balanced_accuracy_score(targets_b, pred_b)

            pred_frames = frame_codec.inverse_transform(pred_a)
            target_frames = frame_codec.inverse_transform(targets_a)
            print (classification_report(target_frames, pred_frames))

            pred_roles = role_codec.inverse_transform(pred_b)
            target_roles = role_codec.inverse_transform(targets_b)
            print (classification_report(target_roles, pred_roles))

            normalize = None # 'true': normalize wrt. the true label count
            conf_a = confusion_matrix(pred_a, targets_a, normalize=normalize, labels=np.arange(2))
            conf_b = confusion_matrix(pred_b, targets_b, normalize=normalize, labels=np.arange(21))

            return  acc_a, acc_b, conf_a, conf_b

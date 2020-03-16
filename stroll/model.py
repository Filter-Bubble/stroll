import sys
import torch
import torch.nn as nn

import dgl.function as fn

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score

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
        return {'h': feats}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.apply_mod = NodeApplyModule(
                self.in_feats,
                self.out_feats,
                self.activation
                )

    def forward(self, g, feature):

        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.mean(msg='m', out='h')

        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class MLP(nn.Module):
    def __init__(
            self,
            in_feats=64,
            out_feats=64,
            activation='relu',
            h_layers=2
            ):
        super(MLP, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.h_layers = h_layers

        layers = []
        for i in range(self.h_layers-1):
            layer = nn.Linear(self.in_feats, self.in_feats)
            nn.init.xavier_uniform_(
                    layer.weight,
                    nn.init.calculate_gain('sigmoid')
                    )
            layers.append(layer)

            layer = nn.BatchNorm1d(self.in_feats)
            layers.append(layer)

            if self.activation == 'relu':
                layer = nn.ReLU()
            elif self.activation == 'tanhshrink':
                layer = nn.Tanhshrink()
            layers.append(layer)

        layer = nn.Linear(self.in_feats, self.out_feats)
        nn.init.xavier_uniform_(
                layer.weight,
                nn.init.calculate_gain('sigmoid')
                )
        layers.append(layer)

        # layers.append(nn.Dropout(p=0.5))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


# https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/4_rgcn.html
# simplify by setting num_bases = num_rels = 3
class RGCN(nn.Module):
    def __init__(
            self,
            in_feats=64,
            out_feats=64,
            activation='relu',
            skip=False
            ):
        super(RGCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.skip = skip

        # weight bases in equation (3)
        self.weight = nn.Parameter(
                torch.Tensor(3, self.in_feats, self.out_feats)
                )
        # nn.init.xavier_uniform_(
        #         self.weight,
        #         gain=nn.init.calculate_gain('relu')
        #         )
        nn.init.kaiming_uniform_(
                self.weight,
                mode='fan_in',
                nonlinearity='relu'
                )

        self.batchnorm = nn.BatchNorm1d(self.out_feats)

        if activation == 'relu':
            self.activation_ = nn.ReLU()
        elif activation == 'tanhshrink':
            self.activation_ = nn.Tanhshrink()
        else:
            print('Activation function not implemented.')
            sys.exit(-1)

    def extra_repr(self):
        return 'in_feats={}, out_feats={}, skip={}'.format(
                self.in_feats, self.out_feats, self.skip
                )

    def forward(self, graph):
        weight = self.weight

        # At each edge, multiply the state h from the source node
        # with a linear weight W_(edge_type)
        def rgcn_msg(edges):
            w = weight[edges.data['rel_type']]
            n = edges.data['norm']
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = torch.bmm(n.reshape(-1, 1, 1), msg.unsqueeze(1)).squeeze()

            return {'m': msg}

        # At each node, we want the summed messages W_(edge_type) \dot h
        # from the incomming edges
        rgcn_reduce = fn.sum(msg='m', out='Swh')

        # Apply activation to the sum(in_edges) W_(edge_type) \dot h
        # TODO: add bias?
        def rgcn_apply(nodes):
            h = nodes.data.pop('h')
            Swh = nodes.data.pop('Swh')

            if self.skip:
                h = self.batchnorm(h + Swh)
            else:
                h = self.batchnorm(h)

            h = self.activation_(h + Swh)

            return {'h': h}

        graph.update_all(rgcn_msg, rgcn_reduce, rgcn_apply)

        return graph


class Net(nn.Module):
    def __init__(
            self,
            in_feats=16,
            h_layers=2,
            h_dims=16,
            out_feats_a=2,
            out_feats_b=16,
            activation='relu'
            ):
        super(Net, self).__init__()
        self.h_layers = h_layers
        self.h_dims = h_dims
        self.in_feats = in_feats
        self.out_feats_a = out_feats_a
        self.out_feats_b = out_feats_b
        self.activation = activation

        layers = []

        # Linear transform of one-hot-encoding to internal representation
        layer = nn.Linear(self.in_feats, self.h_dims)
        nn.init.xavier_uniform_(
                layer.weight,
                nn.init.calculate_gain('relu')
                )
        layers.append(layer)

        # Batchnorm
        layer = nn.BatchNorm1d(self.h_dims)
        layers.append(layer)

        # Activation
        if self.activation == 'relu':
            layer = nn.ReLU()
        elif self.activation == 'tanhshrink':
            layer = nn.Tanhshrink()
        layers.append(layer)

        self.embedding = nn.Sequential(*layers)

        # Hidden layers, each of h_dims to h_dims
        rgcn_layers = []
        for i in range(self.h_layers):
            rgcn_layers.append(
                    RGCN(
                        in_feats=self.h_dims,
                        out_feats=self.h_dims,
                        activation=self.activation,
                        skip=True
                        )
                    )
        self.rgcn = nn.Sequential(*rgcn_layers)

        # a MLP per task
        self.task_a = MLP(
                in_feats=self.h_dims, out_feats=out_feats_a, h_layers=2
                )
        self.task_b = MLP(
                in_feats=self.h_dims, out_feats=out_feats_b, h_layers=2
                )

        # Weight factors for combining the two losses
        self.loss_a = torch.nn.Parameter(torch.tensor([0.]))
        self.loss_b = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, g):
        # Linear transform of one-hot-encoding to internal representation
        g.ndata['h'] = self.embedding(g.ndata['v'])

        # Hidden layers, each of h_dims to h_dims
        g = self.rgcn(g)

        # MLP output
        x_a = self.task_a(g.ndata['h'])
        x_b = self.task_b(g.ndata['h'])

        return x_a, x_b

    def evaluate(self, g):
        self.eval()
        with torch.no_grad():
            logits_F, logits_R = self(g)

            _, pred_F = torch.max(logits_F, dim=1)
            _, pred_R = torch.max(logits_R, dim=1)

            targets_F = g.ndata['frame']
            targets_R = g.ndata['role']

            acc_F = balanced_accuracy_score(targets_F, pred_F)
            acc_R = balanced_accuracy_score(targets_R, pred_R)

            pred_frames = frame_codec.inverse_transform(pred_F)
            target_frames = frame_codec.inverse_transform(targets_F)
            print(classification_report(target_frames, pred_frames))

            pred_roles = role_codec.inverse_transform(pred_R)
            target_roles = role_codec.inverse_transform(targets_R)
            print(classification_report(target_roles, pred_roles))

            normalize = 'true'  # 'true': normalize wrt. the true label count
            conf_F = 100. * confusion_matrix(
                    pred_F, targets_F,
                    normalize=normalize, labels=np.arange(2)
                    )
            conf_R = 100. * confusion_matrix(
                    pred_R, targets_R,
                    normalize=normalize, labels=np.arange(21)
                    )

            return acc_F, acc_R, conf_F, conf_R

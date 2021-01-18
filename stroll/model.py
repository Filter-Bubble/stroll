import sys
import torch
import torch.nn as nn

import dgl.function as fn

from .labels import role_codec, frame_codec


class Embedding(nn.Module):
    """Linear -> BatchNorm -> Activation"""
    def __init__(
            self,
            in_feats=64,
            out_feats=64,
            activation='relu',
            batchnorm=True
            ):
        super(Embedding, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.batchnorm = batchnorm

        layers = []

        layer = nn.Linear(self.in_feats, self.out_feats)
        nn.init.kaiming_uniform_(
                layer.weight,
                mode='fan_in',
                nonlinearity='relu'
                )
        layers.append(layer)

        if self.batchnorm:
            layer = nn.BatchNorm1d(self.out_feats)
            layers.append(layer)

        if self.activation == 'relu':
            layer = nn.ReLU()
        elif self.activation == 'tanhshrink':
            layer = nn.Tanhshrink()
        else:
            print('Activation function not implemented.')
            sys.exit(-1)
        layers.append(layer)

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    """[Linear -> BatchNorm -> Activation] x (n-1) -> Linear"""
    def __init__(
            self,
            in_feats=64,
            out_feats=64,
            activation='relu',
            h_layers=2,
            batchnorm=True,
            pyramid=False,
            bias=True
            ):
        super(MLP, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.h_layers = h_layers
        self.batchnorm = batchnorm
        self.pyramid = pyramid

        if pyramid:
            delta_dims = (self.in_feats - self.out_feats) // self.h_layers
        else:
            delta_dims = 0
        dims_remaining = self.in_feats

        # 10 -> 2 in 2 layers
        # delta = (10 - 2) // 2 =  8 // 2 = 4
        # 10 -> 6 -> 2
        # 211 -> 1 in 2 layers
        # delta = (211 - 1) // 2 = 105
        # 211 -> 106 -> 1

        layers = []
        for i in range(self.h_layers-1):
            layer = nn.Linear(dims_remaining, dims_remaining - delta_dims,
                              bias=bias)
            dims_remaining -= delta_dims
            nn.init.kaiming_uniform_(
                    layer.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                    )
            layers.append(layer)

            if self.batchnorm:
                layer = nn.BatchNorm1d(self.in_feats)
                layers.append(layer)

            if self.activation == 'relu':
                layer = nn.ReLU()
            elif self.activation == 'tanhshrink':
                layer = nn.Tanhshrink()
            else:
                print('Activation function not implemented.')
                sys.exit(-1)
            layers.append(layer)

        layer = nn.Linear(dims_remaining, self.out_feats, bias=bias)
        nn.init.kaiming_uniform_(
                layer.weight,
                mode='fan_in',
                nonlinearity='relu'
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


# https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/4_rgcn.html
# simplify by setting num_bases = num_rels = 3
class RGCNGRU(nn.Module):
    def __init__(
            self,
            in_feats=64,
            out_feats=64,
            num_layers=2
            ):
        super(RGCNGRU, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers

        # weight bases in equation (3)
        self.weight = nn.Parameter(
                torch.Tensor(3, self.in_feats, self.out_feats)
                )
        nn.init.kaiming_uniform_(
                self.weight,
                mode='fan_in',
                nonlinearity='relu'
                )

        self.gru = nn.GRU(
                input_size=self.in_feats,
                hidden_size=self.out_feats,
                num_layers=1,  # for stacked GRU's, not our use case
                bias=True,
                dropout=0,  # we'll use Batchnorm instead
                )

        self.batchnorm = nn.BatchNorm1d(self.out_feats)

    def extra_repr(self):
        return 'in_feats={}, out_feats={}'.format(
                self.in_feats, self.out_feats
                )

    def forward(self, graph):
        weight = self.weight

        # At each edge, multiply the state h from the source node
        # with a linear weight W_(edge_type)
        def rgcn_msg(edges):
            w = weight[edges.data['rel_type']]
            n = edges.data['norm']
            msg = torch.bmm(edges.src['output'].unsqueeze(1), w).squeeze()
            msg = torch.bmm(n.reshape(-1, 1, 1), msg.unsqueeze(1)).squeeze()

            return {'m': msg}

        # At each node, we want the summed messages W_(edge_type) \dot h
        # from the incomming edges
        rgcn_reduce = fn.sum(msg='m', out='Swh')

        # Apply GRU to the sum(in_edges) W_(edge_type) \dot h
        def rgcn_apply(nodes):
            # Shape of h: [len(graph), self.out_feats]
            # GRU wants: [seq_len, batch, input_size]
            output, h_next = self.gru(
                    nodes.data.pop('Swh').view(1, len(graph), self.out_feats),
                    nodes.data.pop('h').view(1, len(graph), self.out_feats)
                    )

            return {
                    'h': h_next.view(len(graph), self.out_feats),
                    'output': output.view(len(graph), self.out_feats)
                    }

        # the embedded node features are the first input to the GRU layer
        graph.ndata['output'] = graph.ndata.pop('h')

        # initial hidden state of the GRU cell
        graph.ndata['h'] = torch.zeros([len(graph), self.out_feats])

        # each step will take the output and hidden state of t-1,
        # and create a new output and hidden state for step t
        for l in range(self.num_layers):
            graph.update_all(rgcn_msg, rgcn_reduce, rgcn_apply)

        # Batchnorm
        graph.ndata.pop('h')
        output = graph.ndata.pop('output')
        graph.ndata['h'] = self.batchnorm(output)

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

        # Embedding
        self.embedding = Embedding(
                in_feats=self.in_feats,
                out_feats=self.h_dims
                )

        # Hidden layers, each of h_dims to h_dims
        self.kernel = RGCNGRU(
                in_feats=self.h_dims,
                out_feats=self.h_dims,
                num_layers=self.h_layers
                )

        # a MLP per task
        self.task_a = MLP(
                in_feats=self.h_dims,
                out_feats=out_feats_a,
                h_layers=2
                )
        self.task_b = MLP(
                in_feats=self.h_dims,
                out_feats=out_feats_b,
                h_layers=2
                )

        # Weight factors for combining the two losses
        self.loss_a = torch.nn.Parameter(torch.tensor([0.]))
        self.loss_b = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, g):
        # Linear transform of one-hot-encoding to internal representation
        g.ndata['h'] = self.embedding(g.ndata['v'])

        # Hidden layers, each of h_dims to h_dims
        g = self.kernel(g)

        # MLP output
        x_a = self.task_a(g.ndata['h'])
        x_b = self.task_b(g.ndata['h'])

        return x_a, x_b

    def label(self, gs):
        logitsf, logitsr = self(gs)
        logitsf = torch.softmax(logitsf, dim=1)
        logitsr = torch.softmax(logitsr, dim=1)

        frame_chance, frame_labels = torch.max(logitsf, dim=1)
        role_chance, role_labels = torch.max(logitsr, dim=1)
        frame_labels = frame_codec.inverse_transform(frame_labels)
        role_labels = role_codec.inverse_transform(role_labels)
        return frame_labels, role_labels, frame_chance, role_chance

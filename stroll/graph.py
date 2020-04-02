import matplotlib.pyplot as plt
import networkx as nx
import torch
import dgl

from .conllu import ConlluDataset

from .labels import upos_codec, xpos_codec, deprel_codec, feats_codec

RID_DIMS = 4


def draw_graph(graph):
    # label_tensor = graph.ndata['form']

    ng = graph.to_networkx()
    # NOTE: for nx the label == the node identifier.
    # nx.relabel_nodes(ng,
    #         lambda x: "{}:{}".format(x, tensor_to_string(label_tensor[x])),
    #         copy=False)

    nx.draw(ng, with_labels=True)
    plt.show()


class GraphDataset(ConlluDataset):
    def __init__(self, filename, features=['UPOS'], sentence_encoder=None):
        super().__init__(filename)
        self.sentence_encoder = sentence_encoder
        self.features = features

        self.in_feats = 0
        if 'UPOS' in features:
            self.in_feats = self.in_feats + len(upos_codec.classes_)
        if 'XPOS' in features:
            self.in_feats = self.in_feats + len(xpos_codec.classes_)
        if 'FEATS' in features:
            self.in_feats = self.in_feats + len(feats_codec.classes_)
        if 'DEPREL' in features:
            self.in_feats = self.in_feats + len(deprel_codec.classes_)
        if 'WVEC' in features:
            self.in_feats = self.in_feats + self.sentence_encoder.dims
        if 'RID' in features:
            self.in_feats = self.in_feats + RID_DIMS

    def __iter__(self):
        for i in range(len(self.sentences)):
            yield self[i]

    def __getitem__(self, index):
        sentence = super().__getitem__(index).encode(
                sentence_encoder=self.sentence_encoder
                )

        g = dgl.DGLGraph()

        # used to map the word ID to the node ID
        wid_to_nid = {}

        # children count per wid
        children_per_wid = {}

        # add nodes
        for token in sentence:
            # add a relative position to the token
            # we label tokens for each head by their order in the sentence
            # ie. first-child, second-child, etc
            if token.ID in children_per_wid:
                children_per_wid[token.ID] += 1
                if children_per_wid[token.ID] == RID_DIMS:
                    children_per_wid[token.ID] = RID_DIMS - 1
            else:
                children_per_wid[token.ID] = 0
            token.RID = torch.zeros([RID_DIMS])
            token.RID[children_per_wid[token.ID]] = 1.

            g.add_nodes(1, {
                'v': torch.cat(
                    [token[f] for f in self.features],
                    0).view(1, -1),
                'frame': token.FRAME,
                'role': token.ROLE
                })
            wid_to_nid[token.ID] = len(g) - 1

        # add edges: word -> head
        for token in sentence:
            if token.HEAD != '0' and token.HEAD != '_':
                g.add_edges(wid_to_nid[token.ID], wid_to_nid[token.HEAD], {
                    'rel_type': torch.tensor([1])
                    })

        # add 1/(3 * in_degree) as a weight factor
        for token in sentence:
            in_edges = g.in_edges(wid_to_nid[token.ID], form='eid')
            if len(in_edges):
                norm = torch.ones([len(in_edges)]) * \
                        (1.0 / (3.0 * len(in_edges)))
                g.edges[in_edges].data['norm'] = norm

        # add edges, these are self-edges, or reversed dependencies
        # give them a weight of 1/3
        norm = torch.tensor([1.0 / 3.0])
        for token in sentence:
            # word -> word (self edge)
            g.add_edges(wid_to_nid[token.ID], wid_to_nid[token.ID], {
                'rel_type': torch.tensor([0]),
                'norm': norm
                })
            # TODO: tokens with ID's like '38.1' don't have a head.
            if token.HEAD != '0' and token.HEAD != '_':
                # head -> word
                g.add_edges(wid_to_nid[token.HEAD], wid_to_nid[token.ID], {
                    'rel_type': torch.tensor([2]),
                    'norm': norm
                    })

        g.sentence = sentence
        return g

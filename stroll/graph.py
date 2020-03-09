import matplotlib.pyplot as plt
import networkx as nx
import torch
import dgl
import dgl.function as fn

from .conllu import ConlluDataset

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
        super().__init__(filename, features)
        self.sentence_encoder = sentence_encoder

    def __iter__(self):
        for i in range(len(self.sentences)):
            yield self[i]
        
    def __getitem__(self, index):
        sentence = super().__getitem__(index).encode(sentence_encoder=self.sentence_encoder)

        g = dgl.DGLGraph()

        # used to map the word ID to the node ID
        wid_to_nid = {}

        # add nodes
        for token in sentence:
            g.add_nodes(1, {
                'v': torch.cat([token[f] for f in self.features], 0).view(1,-1),
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
                norm = torch.ones([len(in_edges)]) * (1.0 / (3.0 * len(in_edges)))
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

        return g

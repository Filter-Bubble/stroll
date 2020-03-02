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

        # add edges
        for token in sentence:
            # TODO: tokens with ID's like '38.1' don't have a head.
            if token.HEAD != '0' and token.HEAD != '_':
                g.add_edges(wid_to_nid[token.ID], wid_to_nid[token.HEAD], {
                    'v': token.DEPREL 
                    })

        return g

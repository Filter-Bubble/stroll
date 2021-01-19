import torch
import dgl

from torch.utils.data import Dataset
from .conllu import ConlluDataset

from .labels import upos_codec, xpos_codec, deprel_codec, feats_codec, get_dims_for_features

RELATION_TYPE_SELF = torch.tensor([0])
RELATION_TYPE_HEAD = torch.tensor([1])
RELATION_TYPE_CHILD = torch.tensor([2])


class GraphDataset(Dataset):
    def __init__(self,
                 filename=None,
                 features=['UPOS'],
                 sentence_encoder=None,
                 dataset=None
                 ):

        if filename:
            self.dataset = ConlluDataset(filename)
        elif dataset:
            # make a graph dataset from the conllu dataset
            self.dataset = dataset

        self.sentence_encoder = sentence_encoder

        in_feats = get_dims_for_features(features)
        if 'WVEC' in features:
            in_feats += self.sentence_encoder.dims

        self.in_feats = in_feats
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

    def __len__(self):
        return len(self.dataset.sentences)

    def __iter__(self):
        for i in range(len(self.dataset.sentences)):
            yield self.dataset[i]

    def conllu(self, index):
        if isinstance(index, dgl.DGLGraph):
            index = index.ndata['sent_index'][0].item()
        return self.dataset[index]

    def __getitem__(self, index):
        g = dgl.DGLGraph()

        g.sentence = self.dataset[index]
        sentence = g.sentence.encode(
                sentence_encoder=self.sentence_encoder
                )

        # add nodes
        for token in sentence:
            g.add_nodes(1, {
                'v': torch.cat(
                    [token[f] for f in self.features],
                    0).view(1, -1),
                'frame': token.FRAME,
                'role': token.ROLE,
                'sent_index': torch.tensor([index], dtype=torch.int32),
                'token_index': torch.tensor(
                    [sentence.index(token.ID)],
                    dtype=torch.int32
                    )
                })

        # add edges: word -> head
        for token in sentence:
            if token.HEAD != '0' and token.HEAD != '_':
                g.add_edges(
                        sentence.index(token.ID),
                        sentence.index(token.HEAD),
                        {'rel_type': RELATION_TYPE_HEAD}
                        )

        # add 1/(3 * in_degree) as a weight factor
        for token in sentence:
            in_edges = g.in_edges(sentence.index(token.ID), form='eid')
            if len(in_edges):
                norm = torch.ones([len(in_edges)]) * \
                        (1.0 / (3.0 * len(in_edges)))
                g.edges[in_edges].data['norm'] = norm

        # add edges, these are self-edges, or reversed dependencies
        # give them a weight of 1/3
        norm = torch.tensor([1.0 / 3.0])
        for token in sentence:
            # word -> word (self edge)
            g.add_edges(
                    sentence.index(token.ID),
                    sentence.index(token.ID),
                    {'rel_type': RELATION_TYPE_SELF, 'norm': norm}
                    )

            # TODO: tokens with ID's like '38.1' don't have a head.
            if token.HEAD != '0' and token.HEAD != '_':
                # head -> word
                g.add_edges(
                        sentence.index(token.HEAD),
                        sentence.index(token.ID),
                        {'rel_type': RELATION_TYPE_CHILD, 'norm': norm}
                        )
        return g

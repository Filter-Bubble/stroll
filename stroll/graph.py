import torch
import dgl

from .conllu import ConlluDataset

from .labels import upos_codec, xpos_codec, deprel_codec, feats_codec

RID_DIMS = 4


class GraphDataset(ConlluDataset):
    def __init__(self,
                 filename=None,
                 features=['UPOS'],
                 sentence_encoder=None,
                 dataset=None
                 ):
        super().__init__(filename)
        if filename is None and dataset is not None:
            # make a graph dataset from the conllu dataset
            self.sentences = dataset.sentences

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

    def __iter__(self):
        for i in range(len(self.sentences)):
            yield self[i]

    def conllu(self, index):
        if isinstance(index, dgl.DGLGraph):
            index = index.ndata['index'][0].item()
        return super().__getitem__(index)

    def __getitem__(self, index):
        unencoded_sentence = super().__getitem__(index)
        sentence = unencoded_sentence.encode(
                sentence_encoder=self.sentence_encoder
                )

        g = dgl.DGLGraph()

        # add nodes
        for token in sentence:
            g.add_nodes(1, {
                'v': torch.cat(
                    [token[f] for f in self.features],
                    0).view(1, -1),
                'frame': token.FRAME,
                'role': token.ROLE,
                'coref': token.COREF,
                'index': torch.Tensor([index]).long()
                })

        # add edges: word -> head
        for token in sentence:
            if token.HEAD != '0' and token.HEAD != '_':
                g.add_edges(
                        sentence.index(token.ID),
                        sentence.index(token.HEAD),
                        {'rel_type': torch.tensor([1])}
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
                    {'rel_type': torch.tensor([0]), 'norm': norm}
                    )

            # TODO: tokens with ID's like '38.1' don't have a head.
            if token.HEAD != '0' and token.HEAD != '_':
                # head -> word
                g.add_edges(
                        sentence.index(token.HEAD),
                        sentence.index(token.ID),
                        {'rel_type': torch.tensor([2]), 'norm': norm}
                        )
        return g

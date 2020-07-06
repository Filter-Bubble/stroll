import argparse
import logging

import torch
import dgl

from torch.utils.data import DataLoader

from stroll.conllu import ConlluDataset
from stroll.graph import GraphDataset
from stroll.coref import preprocess_sentence, postprocess_sentence

from stroll.model import MentionNet
from stroll.labels import FasttextEncoder

# TODO: always label arguments ArgN as a mention?

parser = argparse.ArgumentParser(
        description='Run a mention detection network'
        )
parser.add_argument(
        'input',
        help='Input file in CoNLL format'
        )
parser.add_argument(
        '--model',
        default='models/mentions.pt',
        help='Trained MentionNet to use',
        )
parser.add_argument(
        '--mmax',
        help='Output file in MMAX format',
        )
parser.add_argument(
        '--output',
        help='Output file in conllu format',
        )


def write_output_mmax(dataset, filename):
    keyfile = open(filename, 'w')

    firstDoc = True
    current_doc = None
    for sentence in dataset:
        if sentence.doc_id != current_doc:
            if firstDoc:
                firstDoc = False
            else:
                keyfile.write('#end document\n')

            current_doc = sentence.doc_id
            keyfile.write('#begin document ({});\n'.format(current_doc))
        else:
            keyfile.write('\n')

        for token in sentence:
            if token.FORM == '':
                # these are from unfolding the coordination clauses, dont print
                if token.COREF != '_':
                    logging.error(
                            'Hidden token has a coref={}'.format(token.COREF)
                            )
                    print(sentence)
                    print()
                continue
            if token.COREF != '_':
                coref = token.COREF
            else:
                coref = '-'
            keyfile.write('{}\t0\t{}\t{}\t{}\n'.format(
                sentence.doc_id, token.ID, token.FORM, coref))

    keyfile.write('#end document\n')
    keyfile.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    # 1. load the MentionNet configuration
    state_dict = torch.load(args.model)
    hyperparams = state_dict.pop('hyperparams')
    if 'WVEC' in hyperparams.features:
        if hyperparams.fasttext:
            sentence_encoder = FasttextEncoder(hyperparams.fasttext)
    else:
        sentence_encoder = None

    # 2. load conll file
    dataset = ConlluDataset(args.input)

    # 3. pre-process the dependency tree to unfold coordination
    #   and convert the gold span based mentions to head-based mentions
    for sentence in dataset:
        preprocess_sentence(sentence)

    # 4. make graphs from the conll dataset
    graphset = GraphDataset(
            dataset=dataset,
            sentence_encoder=sentence_encoder,
            features=hyperparams.features
            )
    graph_loader = DataLoader(
        graphset,
        num_workers=2,
        batch_size=500,
        collate_fn=dgl.batch
        )

    # 5. initialize the network
    net = MentionNet(
        in_feats=graphset.in_feats,
        h_layers=hyperparams.h_layers,
        h_dims=hyperparams.h_dims,
        activation='relu'
        )
    net.load_state_dict(state_dict, strict=False)

    net.eval()

    # 6. score mentions
    entity = 0
    for g in graph_loader:
        xa = net(g)
        _, system = torch.max(xa, dim=1)

        # save mentions in the dataset
        sent_index = g.ndata['sent_index']
        token_index = g.ndata['token_index']
        for s, t, m in zip(sent_index, token_index, system):
            if m:
                # treat every mention as a new entity
                dataset[s][t].COREF = entity
                entity += 1
            else:
                dataset[s][t].COREF = '_'

    # 3. convert head-based mentions to span-based mentions
    for sentence in dataset:
        postprocess_sentence(sentence)

    if args.mmax:
        write_output_mmax(dataset, args.output)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(dataset.__repr__())

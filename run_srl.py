#!/usr/bin/env python3
import torch
import argparse
import sys
import logging

import dgl
from torch.utils.data import DataLoader

from stroll.naf import load_naf_stdin, write_frames_to_naf, write_header_to_naf
from stroll.conllu import ConlluDataset
from stroll.graph import GraphDataset
from stroll.srl import make_frames
from stroll.model import Net
from stroll.labels import BertEncoder, FasttextEncoder
from stroll.labels import frame_codec, role_codec

from progress.bar import Bar

# TODO: default Ctarget is from the opinion module, and does not have a set_id()

parser = argparse.ArgumentParser(description='Semantic Role Labelling. Read data in conll or NAF format, and write results to stdout.')
parser.add_argument(
        '--batch_size',
        dest='batch_size',
        default=50,
        help='Inference batch size.'
        )
parser.add_argument(
        '--model',
        dest='model_name',
        help='Model to evaluate',
        required=True
        )
parser.add_argument(
        '--naf',
        default=False,
        action='store_true',
        help='Input in NAF format from stdin'
        )
parser.add_argument(
        '--dataset',
        help='Input in conll format from file',
        )


def infer(net, loader, batch_size=50):
    predicted_frames = []
    predicted_roles = []

    progbar = Bar('Evaluating', max=len(loader))

    net.eval()
    with torch.no_grad():
        for gs in evalloader:

            frame_labels, role_labels, \
               frame_chance, role_chance = net.label(gs)

            node_offset = 0
            for g in dgl.unbatch(gs):
                sentence = eval_set.conllu(g)
                for i, token in enumerate(sentence):
                    token.ROLE = role_labels[i + node_offset]
                    token.pROLE = role_chance[i + node_offset]

                    token.FRAME = frame_labels[i + node_offset]
                    token.pFRAME = frame_chance[i + node_offset]
                node_offset += len(g)

                # match the predicate and roles by some simple graph traversal
                # rules
                frames, orphans = make_frames(sentence)

                if naf:
                    write_frames_to_naf(naf, frames, sentence)

            progbar.next(batch_size)

    progbar.finish()


if __name__ == '__main__':
    args = parser.parse_args()

    state_dict = torch.load(args.model_name)
    hyperparams = state_dict.pop('hyperparams')

    if 'WVEC' in hyperparams.features:
        if hyperparams.fasttext:
            sentence_encoder = FasttextEncoder(hyperparams.fasttext)
        else:
            sentence_encoder = BertEncoder()
    else:
        sentence_encoder = None

    if args.naf:
        dataset, naf = load_naf_stdin()
    elif args.dataset:
        naf = None
        dataset = ConlluDataset(args.dataset)
    else:
        logging.error('No input, you must use --naf or --dataset.')
        sys.exit(-1)

    eval_set = GraphDataset(
            dataset=dataset,
            sentence_encoder=sentence_encoder,
            features=hyperparams.features
            )
    evalloader = DataLoader(
            eval_set,
            batch_size=args.batch_size,
            num_workers=2,
            collate_fn=dgl.batch
            )

    net = Net(
            in_feats=eval_set.in_feats,
            h_layers=hyperparams.h_layers,
            h_dims=hyperparams.h_dims,
            out_feats_a=2,
            out_feats_b=19,
            activation='relu'
            )
    net.load_state_dict(state_dict)

    infer(net, evalloader, batch_size=50)

    if args.naf:
        write_header_to_naf(naf)
        naf.dump()
    else:
        print(dataset)
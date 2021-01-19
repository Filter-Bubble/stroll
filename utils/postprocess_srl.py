import argparse
import logging

import numpy as np

import dgl

import torch
from torch.utils.data import DataLoader

from stroll.model import Net
from stroll.graph import GraphDataset
from stroll.labels import FasttextEncoder


parser = argparse.ArgumentParser(
        description='Postprocessing on SRL conllu files.'
        )
parser.add_argument(
        '--model',
        default='models/srl.pt',
        help='Stroll model to use'
        )
parser.add_argument(
        'input',
        help='Input conllu file to annotate'
        )
parser.add_argument(
        '--output',
        help='Input output conllu file'
        )


class Frame():
    def __init__(self, token=None, p=0.):
        self.arguments = []
        self.p = p
        if token:
            self.ID = token.ID
            self.FORM = token.FORM
            self.LEMMA = token.LEMMA
        else:
            self.ID = 'ORPHAN'
            self.FORM = '-'
            self.LEMMA = '-'

    def __len__(self):
        return len(self.arguments)

    def add_argument(self, role=None, id=None, text=None, p=0.):
        argument = {'id': id, 'role': role, 'text': text, 'p': p}
        self.arguments.append(argument)

    def __repr__(self):
        string = '{} {} ({}) p={:.2f} nargs={}:\n'.format(
                self.ID, self.FORM, self.LEMMA,
                self.p, len(self.arguments)
                )

        for argument in self.arguments:
            string += '   {} {} ({:.2f}): {}\n'.format(
                    argument['id'], argument['role'],
                    argument['p'], argument['text']
                    )

        return string


def adjacency_matrix(sentence):
    # By mulitplying a position vector by the adjacency matrix,
    # we can do one step along the dependency arc.
    L = np.zeros([len(sentence)]*2, dtype=np.int)
    for token in sentence:
        if token.HEAD == "0":
            continue
        L[sentence.index(token.ID), sentence.index(token.HEAD)] = 1

    return L


def build_sentence_parts(sentence, subtree_ids):
    to_descendants = adjacency_matrix(sentence)
    # to_parent = to_descendants.transpos()

    # Raise the matrix to the len(sentence) power; this covers the
    # case where the tree has maximum depth. But first add the unity
    # matrix, so the starting word, and all words we reach, keep a non-zero
    # value.
    is_descendant = to_descendants + np.eye(len(sentence), dtype=np.int)
    is_descendant = np.linalg.matrix_power(is_descendant, len(sentence))

    # collect the subtrees
    subtrees = {}
    for wid in subtree_ids:
        ids, = np.where(is_descendant[:, sentence.index(wid)] > 0)
        subtrees[wid] = [sentence.tokens[i].FORM for i in ids]

    return subtrees


def find_frame(sentence, id):
    """Find a FRAME for the given word ID
    1. look at the parent, and take that if it is a frame.
    2. look at siblings: go to the parent, and consider all descendants"""

    candidates = []

    # # 1. look at the parent, and take that if it is a frame.
    start = sentence[id]
    if start.HEAD == '0':
        # special case for the sentence's head
        # set its parent to itself, so we will look at all descencents
        # in the next step
        parent = start
    else:
        parent = sentence[start.HEAD]
        if parent.FRAME == 'rel':
            return parent.ID
        else:
            # add the parent as candidate frame
            candidates.append(parent)

    # 2. look at siblings: go to the parent, and consider all descendants
    #    but not itself
    to_descendants = adjacency_matrix(sentence)
    to_descendants[sentence.index(start.ID), sentence.index(parent.ID)] = 0

    sibling_ids, = np.where(to_descendants[:, sentence.index(parent.ID)] > 0)
    for sibling_id in sibling_ids:
        sibling = sentence[sibling_id]
        # allowed:     xcomp, compound:prt, ccomp, cop, obj? parataxis?
        allowed = ['xcomp', 'compound:prt', 'ccomp', 'cop']
        if sibling.DEPREL in allowed:
            if sibling.FRAME == 'rel':
                return sibling.ID
            else:
                candidates.append(sibling)
        else:
            # not allowed: conj, parataxis?, advcl, obl, acl
            pass

    for candidate in candidates:
        if candidate.UPOS in ['VERB', 'AUX']:
            return candidate.ID

    return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    state_dict = torch.load(args.model)
    hyperparams = state_dict.pop('hyperparams')

    if 'WVEC' in hyperparams.features:
        sentence_encoder = FasttextEncoder(hyperparams.fasttext)
    else:
        sentence_encoder = None

    eval_set = GraphDataset(
            args.input,
            sentence_encoder=sentence_encoder,
            features=hyperparams.features
            )
    evalloader = DataLoader(
            eval_set,
            batch_size=50,
            collate_fn=dgl.batch,
            num_workers=2
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

                frames = {}
                arguments = {}
                orphans = Frame()

                for token in sentence:
                    if token.FRAME != '_':
                        frames[token.ID] = Frame(token, p=token.pFRAME)
                    if token.ROLE != '_':
                        arguments[token.ID] = token.ROLE

                # get the sentence part for the arguments
                subtrees = build_sentence_parts(sentence, arguments)

                # Match the role to a frame:
                for wid in arguments:
                    fid = find_frame(sentence, wid)
                    if fid is not None:
                        if fid not in frames:
                            # TODO: this is a candidate frame, add it anyways
                            frames[fid] = Frame(
                                    sentence[fid], sentence[fid].pFRAME
                                    )
                        frames[fid].add_argument(
                                role=arguments[wid],
                                p=sentence[wid].pROLE,
                                id=wid,
                                text=subtrees[wid]
                                )
                    else:
                        orphans.add_argument(
                                role=arguments[wid],
                                p=sentence[wid].pROLE,
                                id=wid,
                                text=subtrees[wid]
                                )

                # simple text output
                print('# sent_id =', sentence.sent_id)
                print('# text = ', sentence.full_text)
                for fid in frames:
                    print(frames[fid])
                if len(orphans) > 0:
                    print(orphans)
                print('\n')

    if args.output:
        with open(args.output, 'w') as f:
            f.write(eval_set.dataset.__repr__())

import logging

import numpy as np

import dgl

import torch
from torch.utils.data import DataLoader

from stroll.model import Net
from stroll.graph import GraphDataset
from stroll.labels import BertEncoder, FasttextEncoder

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

    def add_argument(self, role=None, id=None, ids=None, text=None, p=0.):
        argument = {'id': id, 'ids': ids, 'role': role, 'text': text, 'p': p}
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
    treetext = {}
    treeids = {}
    for wid in subtree_ids:
        ids, = np.where(is_descendant[:, sentence.index(wid)] > 0)
        treetext[wid] = [sentence.tokens[i].FORM for i in ids]
        treeids[wid] = [sentence.tokens[i].ID for i in ids]

    return treetext, treeids


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


def make_frames(sentence):
    frames = {}
    arguments = {}
    orphans = Frame()

    for token in sentence:
        if token.FRAME != '_':
            frames[token.ID] = Frame(token, p=token.pFRAME)
        if token.ROLE != '_':
            arguments[token.ID] = token.ROLE

    # get the sentence part for the arguments
    role_text, role_ids = build_sentence_parts(sentence, arguments)

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
                    ids=role_ids[wid],
                    text=role_text[wid]
                    )
        else:
            orphans.add_argument(
                    role=arguments[wid],
                    p=sentence[wid].pROLE,
                    id=wid,
                    ids=role_ids[wid],
                    text=role_text[wid]
                    )
    return frames, orphans

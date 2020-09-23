import sys
import argparse
import logging

import re
import numpy as np

from stroll.conllu import ConlluDataset, Sentence, Token
from stroll.labels import UPOS, upos_codec

# We always ignore punctuation, as they have the sentence root as head
# For multiple heads per span, we take the first head (ie. with lowest token.ID)
# Spans can be headless at this point; these are always spans conataining only punctuation. Drop them.
# There are mulitple spans with the same head, caused by conj.

ref_start = re.compile('^\((\d+)$')
ref_end = re.compile('^(\d+)\)$')
ref_one = re.compile('^\((\d+)\)$')

parser = argparse.ArgumentParser(
        description='Inspect COREF annotations',
        )
parser.add_argument(
        'input',
        nargs='*',
        help='Input conllu file to inspect'
        )


class Stats:
    def __init__(self):
        self.chain_lengths = np.zeros(500)
        self.mentions_scored = 0
        self.mentions_exact = 0
        self.mentions_longer = 0
        self.mentions_shorter = 0

    def print(self):
        print('mentions_scored', self.mentions_scored)
        print('mentions_exact', self.mentions_exact)
        print('mentions_longer', self.mentions_longer)
        print('mentions_shorter', self.mentions_shorter)
        for i in range(len(self.chain_lengths)):
            print(i, self.chain_lengths[i])


# tree transform             no tree transform
# mentions_scored 280212     mentions_scored 268593
# mentions_exact 256258      mentions_exact 244099
# mentions_longer 15670      mentions_longer 16528
# mentions_shorter 8284      mentions_shorter 7966
# 0 0.0                      0 0.0
# 1 173259.0                 1 165491.0
# 2 14538.0                  2 14132.0
# 3 4702.0                   3 4504.0
# 4 2317.0                   4 2237.0
# 5 1275.0                   5 1250.0
# 6 889.0                    6 849.0
# 7 553.0                    7 520.0
# 8 400.0                    8 394.0
# 9 285.0                    9 267.0
# 10 207.0                   10 206.0
# 11 178.0                   11 171.0
# 12 140.0                   12 126.0
# 13 106.0                   13 93.0
# 14 79.0                    14 77.0
def transform_tree(sentence):
    # find all coordinations, ie. tokens with DEPREL = 'conj'
    coordinations = {}
    for token in sentence:
        if token.DEPREL == 'conj':
            # identify the coordinations by the head
            coordination_id = token.HEAD
            if coordination_id in coordinations:
                coordinations[coordination_id].append(token.ID)
            else:
                coordinations[coordination_id] = [token.ID]

    # transform each coordination
    #           ^                           ^
    #           | deprel                    | deprel
    #         tokenA              =>       tokenX
    #       /conj    \ conj          /conj  |conj  \conj
    #   tokenB        tokenC      tokenA   tokenB   tokenC
    for tokenA_id in coordinations:
        coordination = coordinations[tokenA_id]
        tokenA = sentence[tokenA_id]

        # attach tokens directly to the original HEAD of the first token
        for token_id in coordination:
            sentence[token_id].HEAD = tokenA.HEAD

        # get a span for tokenA
        spans = build_sentence_parts(sentence, [tokenA.ID])
        spanA = spans[tokenA.ID]

        # create a dummy node to represent the full coordination
        tokenX = Token([
            '{}'.format(len(sentence) + 1),  # ID
            '',  # FORM
            '',  # LEMMA
            '_',  # UPOS
            '_',  # XPOS
            '_',  # FEATS
            tokenA.HEAD,
            tokenA.DEPREL,
            tokenA.DEPS,
            tokenA.MISC,
            tokenA.FRAME,
            tokenA.ROLE,
            '_'  # COREF
            ])

        # attach all tokens to the dummy
        for token_id in coordination:
            sentence[token_id].HEAD = tokenX.ID

        tokenA.HEAD = tokenX.ID
        tokenA.DEPREL = 'conj'
        tokenA.FRAME = '_'
        tokenA.ROLE = '_'

        # insert the dummy token to the right of the first token
        # that is not part of span A
        place = sentence.index(tokenA.ID)
        while sentence[place].ID in spanA or sentence[place].UPOS == 'PUNCT':
            place += 1
            if place == len(sentence):
                break

        sentence.tokens.insert(place, tokenX)
        sentence._build_id_to_index()

        # fix remaining issues:
        # - punctuation is a child of the root token
        if tokenX.DEPREL == 'root':
            for token in sentence:
                if token.UPOS == 'PUNCT':
                    token.HEAD = tokenX.ID

    return sentence


def adjacency_matrix(sentence):
    # By mulitplying a position vector by the adjacency matrix,
    # we can do one step along the dependency arc.
    L = np.zeros([len(sentence)]*2, dtype=np.int)
    for token in sentence:
        if token.HEAD == "0" or token.DEPREL == 'parataxis':
            continue
        L[sentence.index(token.ID), sentence.index(token.HEAD)] = 1

    return L


def build_sentence_parts(sentence, subtree_ids):
    """Build the sentence part (mention) corresponding to the given head.

                    full     a-zA-Z  adp-det first head   adp     cc     sconj cconj adp  parataxis
    mentions_scored 285750   285750  285750      285750  285750  285750  285750   285750    285750
    mentions_exact  129262   140372  192650      192662  243251  253291  256356   254707    256883
    mentions_longer 144523   141931   89455       89413   38387   28305   25149    23246     20464
    mentions_shorter 11965     3447    3645        3675    4112    4154    4245     7797      8403
    """
    to_descendants = adjacency_matrix(sentence)

    # Raise the matrix to the len(sentence) power; this covers the
    # case where the tree has maximum depth. But first add the unity
    # matrix, so the starting word, and all words we reach, keep a non-zero
    # value.
    is_descendant = to_descendants + np.eye(len(sentence), dtype=np.int)
    is_descendant = np.linalg.matrix_power(is_descendant, len(sentence))

    # collect the subtrees
    subtrees_form = {}
    subtrees_upos = {}
    for wid in subtree_ids:
        ids, = np.where(is_descendant[:, sentence.index(wid)] > 0)
        # en, ook, als,
        if len(ids) >= 2:
            if sentence.tokens[ids[0]].UPOS in ['ADV', 'CCONJ', 'SCONJ']:
                ids = ids[1:]

        # in, op, voor, bij,
        if len(ids) >= 2:
            if sentence.tokens[ids[0]].UPOS in ['ADP', 'ADV']:
                ids = ids[1:]

        tree = [sentence.tokens[i].ID for i in ids]
        subtrees_form[wid] = tree

    return subtrees_form


def get_mentions(sentence):
    """Build the spans (mentions) from the conll file using the bra-ket notation.
    The span's head is determined as first token with token.HEAD not in [token.ID for token in span]
    """
    spans = []
    for token in sentence:
        refs = token.COREF.split('|')
        for ref in refs:
            ms = ref_start.match(ref)
            me = ref_end.match(ref)
            mo = ref_one.match(ref)

            if ref == '_':
                # not a mention
                continue

            # start of a mention: create a new, open, span
            if ms:
                refid, = ms.groups(0)
                spans.append({
                        'refid': refid,
                        'start': token.ID,
                        'end': '_',
                        'text': '',
                        'heads': [],
                        'ids': []
                        })
            # end of a mention: close the span
            elif me:
                refid, = me.groups(0)
                spanid = len(spans) - 1
                # find the most recent span with this refid, ie. top-of-the-stack
                while spanid >= 0 and spans[spanid]['refid'] != refid:
                    spanid -= 1

                spans[spanid]['end'] = token.ID

            # single word mention
            elif mo:
                refid, = mo.groups(0)
                spans.append({
                        'refid': refid,
                        'start': token.ID,
                        'end': token.ID,
                        'text': '',
                        'heads': [],
                        'ids': []
                        })

        # gather information for spans
        for span in spans:
            if span['end'] in ['_', token.ID]:
                # build the mention text by appending the word to all open spans
                span['text'] += ' ' + token.FORM

                # keep a list of heads and ids to find the head later
                if token.UPOS != 'PUNCT':
                    span['heads'].append(token.HEAD)
                    span['ids'].append(token.ID)

    # find the heads for each span
    mentions_heads = []
    valid_mentions = []
    for span in spans:
        heads = span['heads']
        ids = span['ids']

        # take as head the FIRST token that has a head not part of the span
        heads_found = 0
        for i, head in enumerate(heads):
            if head not in ids:
                if heads_found == 0:  # take the first
                    span['head'] = ids[i]
                heads_found += 1

        # headless mentions can occur for mentions containing only punctuation
        if heads_found != 0:
            if span['head'] in mentions_heads:
                # print('HEAD COLLISION', span['refid'], span['head'])
                pass
            else:
                mentions_heads.append(span['head'])
                valid_mentions.append(span)

        # if heads_found != 1:
        #     print('HEAD COUNT:', heads_found, 'span:', span)


    return valid_mentions


def inspect_dataset(dataset, stats):
    # Count the number of occurences per refid
    ref_count = {}
    for sentence in dataset:
        sentence = transform_tree(sentence)
        mentions = get_mentions(sentence)
        for mention in mentions:
            refid = mention['refid']
            if refid in ref_count:
                ref_count[refid] += 1
            else:
                ref_count[refid] = 1

    # build chain length statistics
    for refid in ref_count:
        stats.chain_lengths[ref_count[refid]] += 1

    # Go over each mention
    for sentence in dataset:
        sentence = transform_tree(sentence)
        mentions = get_mentions(sentence)

        # print('File: ', conllufile)
        # print(sentence)

        heads = [mention['head'] for mention in mentions]
        parts = build_sentence_parts(sentence, heads)

        for mention in mentions:
            # print('id=', mention['refid'],
            #       'count=', ref_count[mention['refid']],
            #       's=', mention['start'],
            #       'e=', mention['end'],
            #       'h=', mention['head']
            #       )
            # print('Span       :', mention['text'])
            part_as_text = ' '.join([ sentence[ID].FORM for ID in parts[mention['head']] ])

            stats.mentions_scored += 1
            sys = re.sub('[^a-zA-Z]', '', part_as_text.replace(' ', ''))
            gold = re.sub('[^a-zA-Z]', '', mention['text'].replace(' ', ''))
            if sys == gold:
                stats.mentions_exact += 1
                # print('From syntax EXACT:  ', part_as_text, parts[mention['head']])
            elif len(sys) < len(gold):
                stats.mentions_shorter += 1
                # print('From syntax SHORTER:', part_as_text, parts[mention['head']])
            else:
                stats.mentions_longer += 1
                # print('From syntax LONGER: ', part_as_text, parts[mention['head']])

        # print('\n\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    if not isinstance(args.input, list):
        args.input = [args.input]

    for conllufile in args.input:
        logging.info('Processing file {}'.format(conllufile))
        dataset = ConlluDataset(conllufile)
        mentions_in_doc = 0
        tokens_in_doc = 0

        transformed = conllufile + '_coref'
        #outputfile = open(transformed, 'w')

        isFirst = True
        for sentence in dataset:
            sentence = transform_tree(sentence)
            mentions = get_mentions(sentence)

            for token in sentence:
                token.COREF_HEAD = '_'

            for mention in mentions:
                sentence[mention['head']].COREF_HEAD = mention['refid']
            logging.info('Number of mentions in sentence {}'.format(len(mentions)))
            mentions_in_doc += len(mentions)
            tokens_in_doc += len(sentence)

            #if isFirst:
            #    outputfile.write(sentence.__repr__())
            #    outputfile.write('\n')
            #    isFirst = False
            #else:
            #    outputfile.write('\n')
            #    outputfile.write(sentence.__repr__())
            #    outputfile.write('\n')

        logging.info('Number of mentions in document {}'.format(mentions_in_doc))
        logging.info('Number of tokens in document {}'.format(tokens_in_doc))
        #outputfile.close()

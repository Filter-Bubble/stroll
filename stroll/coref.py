import re
import numpy as np

from stroll.conllu import Token

# We always ignore punctuation, as they have the sentence root as head
# For multiple heads per span, we take the 1st head (ie. with lowest token.ID)
# Spans can be headless at this point; these are always spans conataining
# only punctuation. Drop them.
# There can be mulitple spans with the same head, take the first occuring.
# Trim the span by removing specific UPOS tokens from the front (ADP, CCONJ,..)

ref_start = re.compile('^\((\d+)$')
ref_end = re.compile('^(\d+)\)$')
ref_one = re.compile('^\((\d+)\)$')


def adjacency_matrix(sentence):
    # By mulitplying a position vector by the adjacency matrix,
    # we can do one step along the dependency arc.
    L = np.zeros([len(sentence)]*2, dtype=np.int)
    for token in sentence:
        if token.HEAD == "0" or token.DEPREL == 'parataxis':
            continue
        L[sentence.index(token.ID), sentence.index(token.HEAD)] = 1

    return L


def build_spans_from_heads(sentence, subtree_ids):
    """Build the sentence part (mention) corresponding to the given head.
                scored exact   longer  shorter
    full        285750 129262  144523  11965
    a-zA-Z      285750 140372  141931  3447
    adp-det     285750 192650  89455   3645
    first head  285750 192662  89413   3675
    adp         285750 243251  38387   4112
    cc          285750 253291  28305   4154
    sconj       285750 256356  25149   4245
    cconj adp   285750 254707  23246   7797
    parataxis   285750 256883  20464   8403
    """
    to_descendants = adjacency_matrix(sentence)

    # Raise the matrix to the len(sentence) power; this covers the
    # case where the tree has maximum depth. But first add the unity
    # matrix, so the starting word, and all words we reach, keep a non-zero
    # value.
    is_descendant = to_descendants + np.eye(len(sentence), dtype=np.int)
    is_descendant = np.linalg.matrix_power(is_descendant, len(sentence))

    # collect the spans
    spans = {}
    for wid in subtree_ids:
        ids, = np.where(is_descendant[:, sentence.index(wid)] > 0)
        # en, ook, als,
        if len(ids) >= 2:
            if sentence[ids[0]].UPOS in ['ADV', 'CCONJ', 'SCONJ']:
                ids = ids[1:]

        # in, op, voor, bij,
        if len(ids) >= 2:
            if sentence[ids[0]].UPOS in ['ADP', 'ADV']:
                ids = ids[1:]

        spans[wid] = {
                'start': sentence[ids[0]].ID,
                'end': sentence[ids[-1]].ID,
                'ids': [sentence[i].ID for i in ids]
                }

    return spans


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
        spans = build_spans_from_heads(sentence, [tokenA.ID])
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
        while (
         sentence[place].ID in spanA['ids'] or sentence[place].UPOS == 'PUNCT'
         ):
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


def get_mentions_from_bra_ket(sentence):
    """Build the spans (mentions) from the conll file using the bra-ket notation.
    The span's head is determined as first token with token.HEAD not
    in [token.ID for token in span]
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
                # find the most recent span with this refid,
                # ie. top-of-the-stack
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
                # build the mention text by appending the word to
                # all open spans
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


def preprocess_sentence(sentence):
    """Preprocess a Sentence containing coref annotation in bra-ket notation.

    Modify the coordination clauses in the dependency tree
    Replace span based mentions to syntactic head based mentions
    """
    sentence = transform_tree(sentence)
    mentions = get_mentions_from_bra_ket(sentence)

    for token in sentence:
        token.COREF = '_'

    for mention in mentions:
        sentence[mention['head']].COREF = mention['refid']

    return sentence


def postprocess_sentence(sentence):
    """Postprocess a Sentence containing syntactic head based mentions.

    Replace the syntactic head based mentions by spans.
    """
    heads = []
    head_to_entity = {}
    for token in sentence:
        if token.COREF != '_':
            heads.append(token.ID)
            head_to_entity[token.ID] = token.COREF
            token.COREF = '_'

    spans = build_spans_from_heads(sentence, heads)
    for head in spans:
        span = spans[head]
        entity = head_to_entity[head]

        astart = sentence[span['start']].COREF
        aend = sentence[span['end']].COREF

        if span['start'] == span['end']:
            # add a single token mention
            if astart == '_':
                astart = '({})'.format(entity)
            else:
                astart += '|({})'.format(entity)
        else:
            # add a multi token mention
            if astart == '_':
                astart = '({}'.format(entity)
            else:
                astart += '|({}'.format(entity)

            if aend == '_':
                aend = '{})'.format(entity)
            else:
                aend += '|{})'.format(entity)

            sentence[span['end']].COREF = aend
        sentence[span['start']].COREF = astart

    return sentence

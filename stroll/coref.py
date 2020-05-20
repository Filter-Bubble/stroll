import re
import numpy as np

import logging

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

COPULA_NOUN_DESC_MOVE_TO_VERB = [
        'advcl', 'advmod', 'aux', 'aux:pass', 'case', 'cc', 'csubj', 'expl',
        'expl:pv', 'iobj', 'mark', 'nsubj', 'obl', 'obl:agent', 'orphan',
        'parataxis', 'punct'
        ]

ADJACENCY_SKIP_DEPREL = [
        'parataxis', 'punct'
        ]

MENTION_REMOVE_DESC = [
        'advcl', 'aux:pass', 'expl:pv', 'aux', 'obl:agent', 'csubj', 'orphan',
        'nsubj', 'obl', 'mark', 'advmod', 'ccomp', 'parataxis', 'iobj', 'expl',
        'cc', 'case', 'punct'
        ]


class Mention():
    """A Mention

      sentence The corresponding Sentence
      head     The ID of the mention head
      refid    The entity this mention refers to (integer)

      The span corresponding to this mention is derived from the head,
      but processing has been done to remove syntax words.

      start    The ID of the first token of the mention
      end      The ID of the last token of the mention
      ids      The IDs of all tokens part of the mention
    """

    def __init__(self,
                 sentence=None,  # The corresponding Sentence
                 head=None,  # The ID of the mention head
                 refid=None,  # The entity this mention refers to
                 start=None,  # The ID of the first token of the mention
                 end=None,  # The ID of the last token of the mention
                 ids=None  # The IDs of all tokens part of the mention
                 ):
        self.sentence = sentence
        self.head = head
        self.refid = refid
        self.start = start
        self.end = end
        self.ids = ids

    def type(self):
        """The type is one of LIST, PRONOMIAL, PROPER, NOMINAL"""
        sentence = self.sentence
        token = sentence[self.head]
        if token.FORM == '':
            mtype = 'LIST'
        elif token.UPOS == 'PRON':
            mtype = 'PRONOMIAL'
        elif token.UPOS == 'PROPN':
            mtype = 'PROPER'
        else:
            mtype = 'NOMINAL'
        return mtype

    def nested(mention):
        """A mention is nested if any of its parents are a mention."""
        sentence = mention.sentence

        token = sentence[mention.head]
        while token.HEAD != '0':
            token = sentence[token.HEAD]
            if token.COREF != '_':
                return 1.0

        return 0.0


def mentions_heads_agree(mentionA, mentionB):
    """The (uncased) head.FORM of the mentions match"""

    # single token mentions, just compare them
    if len(mentionA.ids) == 1 and len(mentionB.ids) == 1:
        headA = mentionA.sentence[mentionA.head].FORM.lower()
        headB = mentionB.sentence[mentionB.head].FORM.lower()

        if headA == headB:
            return 1.0
        return 0

    # multi-token: there can be multi-word-expresions to deal with
    # we only resolve the 'flat' deprel: 'John <-flat- Johnson'
    # and match when metionA is contained in B or B in A.

    # build the multi word string for mention A
    headA_contents = []
    sentA = mentionA.sentence

    headA_contents.append(sentA[mentionA.head].FORM.lower())
    idx = sentA.index(mentionA.head) + 1
    while sentA[idx].DEPREL == 'flat':
        headA_contents.append(sentA[idx].FORM.lower())
        idx += 1

    # build the multi word string for mention B
    headB_contents = []
    sentB = mentionB.sentence

    headB_contents.append(sentB[mentionB.head].FORM.lower())
    idx = sentB.index(mentionB.head) + 1
    while sentB[idx].DEPREL == 'flat':
        headB_contents.append(sentB[idx].FORM.lower())
        idx += 1

    # All tokens from the shortest one should be in the largest
    if len(headA_contents) < len(headB_contents):
        for form in headA_contents:
            if form not in headB_contents:
                return 0.0
        return 1.0
    else:
        for form in headB_contents:
            if form not in headA_contents:
                return 0.0
    return 1.0


def mentions_match_exactly(mentionA, mentionB):
    """The (uncased) sentence[ids].FORM of the mentions match"""
    if len(mentionA.ids) != len(mentionB.ids):
        return 0.0

    sentA = mentionA.sentence
    sentB = mentionB.sentence
    for i in range(len(mentionA.ids)):
        formA = sentA[mentionA.ids[i]].FORM
        formB = sentB[mentionB.ids[i]].FORM
        if formA.lower() != formB.lower():
            return 0.0

    return 1.0


def mentions_match_relaxed(mentionA, mentionB):
    """
    All content (NOUNs and PROPNs) words of one are included in the
    other mention.
    """
    sentA = mentionA.sentence
    sentB = mentionB.sentence

    contentA = []
    for id in mentionA.ids:
        token = sentA[id]
        if token.UPOS in ['PROPN', 'NOUN']:
            contentA.append(token.FORM.lower())

    for id in mentionB.ids:
        token = sentB[id]
        if token.UPOS in ['PROPN', 'NOUN']:
            if token.FORM.lower() in contentA:
                return 1.0

    return 0.0


def mentions_are_overlapping(mentionA, mentionB):
    """
    The mentions are from the same sentence, and overlap in the sentence.
    """
    sentA = mentionA.sentence
    sentB = mentionB.sentence

    if sentA.doc_id != sentB.doc_id:
        return 0.0
    if sentA.rank != sentB.rank:
        return 0.0

    if mentionA.head in mentionB.ids or mentionB.head in mentionA.ids:
        return 1.0

    return 0.0


def adjacency_matrix(sentence):
    """
    Build a modified adjacency matrix from the sentence.

    By mulitplying a position vector by the adjacency matrix,
    we do one step along the dependency arcs.
    This is used for constructing the mention text from a head, so not all arcs
    are useful. At the moment parataxis and punctuation arcs are ignored.
    """
    L = np.zeros([len(sentence)]*2, dtype=np.int)
    for token in sentence:
        if token.HEAD == "0" or token.DEPREL in ADJACENCY_SKIP_DEPREL:
            continue
        L[sentence.index(token.ID), sentence.index(token.HEAD)] = 1

    return L


def build_mentions_from_heads(sentence, heads):
    """
    Build the mention text corresponding to the given head by
    gathering leaves from the subtree.  The subtree is pruned by removing all
    direct descendants with a DEPREL in a black list, MENTION_REMOVE_DESC.
    This list is based on statistics over the SoNaR dataset.

    Returns a dict of Mention, indexed by the Token.ID of the mention head.
    """
    to_descendants = adjacency_matrix(sentence)

    # Raise the matrix to the len(sentence) power; this covers the
    # case where the tree has maximum depth. But first add the unity
    # matrix, so the starting word, and all words we reach, keep a non-zero
    # value.
    is_descendant = to_descendants + np.eye(len(sentence), dtype=np.int)
    is_descendant = np.linalg.matrix_power(is_descendant, len(sentence))

    # collect the spans
    mentions = {}
    for wid in heads:
        # get all tokens that make up the subtree, by construction
        ids, = np.where(is_descendant[:, sentence.index(wid)] > 0)

        # prune direct descendants at this point by looking at the deprel to
        # our subtree root to deal with leading 'en, ook, als, in, op, voor,
        # bij, ...' that are not part of the mention.
        # this works better than removing [ADV, CCONJ, SCONJ], and
        # then [ADP, ADV] from the front.
        # Also remove empty tokens (could have been added by our preprocesing)
        ids_to_prune = []
        for token in sentence:
            if token.HEAD == wid and token.DEPREL in MENTION_REMOVE_DESC:
                prune, = np.where(
                        is_descendant[:, sentence.index(token.ID)] > 0
                        )
                ids_to_prune += list(prune)

        pruned_ids = []
        for id in ids:
            if id not in ids_to_prune and sentence[id].FORM != '':
                pruned_ids.append(id)

        mentions[wid] = Mention(
                head=wid,
                sentence=sentence,
                refid=sentence[wid].COREF,
                start=sentence[pruned_ids[0]].ID,
                end=sentence[pruned_ids[-1]].ID,
                ids=[sentence[i].ID for i in pruned_ids]
                )

    return mentions


def get_mentions(sentence):
    """
    Return a list of Mention objects, from the annotation in the sentence.

    NOTE: first modify the sentence with transform_tree
    """
    heads = []
    for token in sentence:
        if token.COREF != '_':
            heads.append(token.ID)
    return build_mentions_from_heads(sentence, heads)


def transform_coordinations(sentence):
    """
    Transform UD coordination to be more like SD coordination.

    Universal Dependency style coordination is not suitable for our head-based
    approach; in the text '.. A and B ..' the possible heads are:

    A which will correspond to the text 'A and B'
    B which will correspond to 'and B'.

    Add an extra node, and transform the dependency graph, to allow selecting:
    'A', 'and B', 'A and B'
    """

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
        mentions = build_mentions_from_heads(sentence, [tokenA.ID])
        mention = mentions[tokenA.ID]

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
         sentence[place].ID in mention.ids or sentence[place].DEPREL == 'punct'
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
                if token.DEPREL == 'punct':
                    token.HEAD = tokenX.ID

    return sentence


def swap_copula(sentence, noun, verb):
    """
    We promote the copula token to be the head (like other verbs would be),
    and attach the mention to it.  We then try to clean up the graph by moving
    a number of descendants of the old head (ie part the mention) to the
    copula. The choice is based on the DEPREL, for a list see
    COPULA_NOUN_DESC_MOVE_TO_VERB.
    """
    # 1. swap the deprel of the 'cop' and its syntactic head
    # (lets call it its noun)
    verb.DEPREL = noun.DEPREL
    noun.DEPREL = 'cop'
    verb.HEAD = noun.HEAD
    noun.HEAD = verb.ID

    # 2. of the dependents of the noun, move some to the verb
    for token in sentence:
        if token.HEAD == noun.ID:
            if token.DEPREL in COPULA_NOUN_DESC_MOVE_TO_VERB:
                token.HEAD = verb.ID

    return sentence


def transform_copulas(sentence):
    """
    Transform the UD copola usage.

    We focus on Dutch, which has explicit copolas: 'Dat is fantastisch.'
    UD requires 'Dat' to be the head, which leads to mentions spanning the
    whole sentence, which in turn will confuse all our mention and pairwise
    mention features (nesting, matching).

    For cases with multiple linked copulas 'Dat is en blijft fantastisch',
    we set the DEPREL to 'conj', like UD does for normal verbs.
    """
    # 1. gather all 'cop' that point to another 'cop'
    copulas = []
    chained_copulas = []
    for token in sentence:
        if token.DEPREL != 'cop':
            continue
        if token.HEAD == '0':
            # Inconceivable! DEPREL == 'cop', but HEAD == '0'
            return sentence

        if sentence[token.HEAD].DEPREL == 'cop':
            # we have a -cop-> b -cop-> ..
            # turn this into a conjunction
            chained_copulas.append(token)
        else:
            copulas.append(token)

    # if we didnt find any copulas we're done
    if len(copulas) == 0:
        return sentence

    # 2. change the releation to conj
    for token in chained_copulas:
        token.DEPREL = 'conj'

    # 3. swap the noun and verb
    for verb in copulas:
        noun = sentence[verb.HEAD]
        swap_copula(sentence, noun, verb)

    # 4. Clean-up:, all 'punct' must point to the head
    for token in sentence:
        if token.HEAD == '0':
            # found the head
            head_id = token.ID
            break

    for token in sentence:
        if token.DEPREL == 'punct':
            token.HEAD = head_id

    return sentence


def transform_tree(sentence):
    """
    Make dependency graph better suited for our task.
    This calls: transform_copulas and transform_coordinations.
    """
    sentence = transform_copulas(sentence)
    sentence = transform_coordinations(sentence)
    return sentence


def get_spans_from_brak_ket(sentence):
    """
    Get spans from the sentence' brak-ket coref annotation.

    Return spans as a list of dicts:
       refid    mention id
       start    first Token.ID of the mention
       end      last Token.ID of the mention
       heads    list of Token.HEAD
       ids      list of Token.ID

    Where the list is over tokens that are gramatically part of the sentence.
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
                        'heads': [],
                        'ids': []
                        })
            # end of a mention: close the span
            elif me:
                refid, = me.groups(0)
                spanid = len(spans) - 1
                # find the most recent span with this refid,
                # ie. top-of-the-stack
                while spanid >= 0 and (
                        spans[spanid]['refid'] != refid or
                        spans[spanid]['end'] != '_'
                        ):
                    spanid -= 1

                spans[spanid]['end'] = token.ID

            # single word mention
            elif mo:
                refid, = mo.groups(0)
                spans.append({
                        'refid': refid,
                        'start': token.ID,
                        'end': token.ID,
                        'heads': [],
                        'ids': []
                        })

        # gather information for spans
        for span in spans:
            if span['end'] in ['_', token.ID]:
                # keep a list of heads and ids to find the head later
                if token.DEPREL != 'punct':
                    span['heads'].append(token.HEAD)
                    span['ids'].append(token.ID)

    # sanity check: are all spans closed?
    for span in spans:
        if span['end'] == '_':
            logging.error('Unclosed span found {} {} {}'.format(
                sentence.doc_id, sentence.sent_id, span['refid'])
                )

    return spans


def get_mentions_from_bra_ket(sentence):
    """
    Build the mentions from the conll file using the bra-ket notation.
    The span's head is determined as first token with token.HEAD not
    in [token.ID for token in span]

    Returns a list of Mention
    """
    spans = get_spans_from_brak_ket(sentence)

    # find the heads for each span
    mentions_heads = []
    valid_mentions = []
    for span in spans:
        heads = span['heads']
        ids = span['ids']

        # take as head the FIRST token that has a head not part of the span
        head_found = False
        for i, head in enumerate(heads):
            if head not in ids:
                span['head'] = ids[i]
                head_found = True
                break

        # headless mentions can occur for mentions containing only punctuation
        if head_found:
            if span['head'] in mentions_heads:
                # Two spans claim the same head; conceptually something like
                # 1 John      ....  ..     (0|(0)
                # 2 Johnson   flat   1     0)
                # for now, take the longer span, as it carries more information
                replace_mention = False
                for s in valid_mentions:
                    if s['head'] == span['head']:
                        if len(s['ids']) > len(ids):
                            replace_mention = s
                            break

                if replace_mention:
                    valid_mentions.remove(replace_mention)
                    valid_mentions.append(span)
            else:
                mentions_heads.append(span['head'])
                valid_mentions.append(span)

    return valid_mentions


def preprocess_sentence(sentence):
    """
    Preprocess a Sentence containing coref annotation in bra-ket notation.

    Modify the coordination and copula in the dependency tree
    Replace span based mentions with syntactic head based mentions
    """
    sentence = transform_tree(sentence)
    mentions = get_mentions_from_bra_ket(sentence)

    for token in sentence:
        token.COREF = '_'

    for mention in mentions:
        sentence[mention['head']].COREF = mention['refid']

    return sentence


def postprocess_sentence(sentence):
    """
    Postprocess a Sentence containing syntactic head based mentions.

    Replace the syntactic head based mentions by spans.
    Do *NOT* undo changes made to the dependency graph.
    """
    heads = []
    head_to_entity = {}
    for token in sentence:
        if token.COREF != '_':
            heads.append(token.ID)
            head_to_entity[token.ID] = token.COREF
            token.COREF = '_'

    mentions = build_mentions_from_heads(sentence, heads)
    # mentions is a dict of {head]} -> mention
    for head in mentions:
        mention = mentions[head]
        entity = head_to_entity[head]

        astart = sentence[mention.start].COREF
        aend = sentence[mention.end].COREF

        if mention.start == mention.end:
            # add a single token mention
            if astart == '_':
                astart = '({})'.format(entity)
            else:
                astart = '{}|({})'.format(astart, entity)
        else:
            # add a multi token mention
            if astart == '_':
                astart = '({}'.format(entity)
            else:
                astart = '({}|{}'.format(entity, astart)

            if aend == '_':
                aend = '{})'.format(entity)
            else:
                aend = '{})|{}'.format(entity, aend)

            sentence[mention.end].COREF = aend
        sentence[mention.start].COREF = astart

    return sentence

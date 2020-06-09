import re

import numpy as np

import torch

import logging

from stroll.conllu import Token
from stroll.labels import to_one_hot, mention_type_codec

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
        'punct'
        ]

MENTION_REMOVE_DESC = [
        'advcl', 'aux:pass', 'expl:pv', 'aux', 'obl:agent', 'csubj', 'orphan',
        'nsubj', 'obl', 'mark', 'advmod', 'ccomp', 'parataxis', 'iobj', 'expl',
        'cc', 'case', 'punct'
        ]

MENTION_REMOVE_LEADING = [
        'ADP', 'ADV', 'CCONJ', 'SCONJ'
        ]


class Mention():
    """
    A Mention

    Properties:
      sentence The corresponding Sentence
      head      The token.ID of the mention head
      refid     The entity this mention refers to
      start     The ID of the first token of the mention
      end       The ID of the last token of the mention
      ids       The IDs of all tokens part of the mention

      NOTE: The span corresponding to this mention is derived from
      the head, but processing has been done to remove syntax words.
      They are basically the min(ids) and max(ids).  If the language
      allows discontinuities, the set of ids is a better representation.

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
        if ids:
            self.ids = ids
        else:
            self.ids = []

    def __repr__(self):
        p = ""
        p += '# sent_id:   {}\n'.format(self.sentence.sent_id)
        p += '# full_text: {}\n'.format(self.sentence.full_text)
        p += 'refid= {}\n'.format(self.refid)
        p += 'head=  {}\n'.format(self.head)
        p += 'span=  {}-{}\n'.format(self.start, self.end)
        return p

    def type(self):
        """The type is one of LIST, PRONOMIAL, PROPER, NOMINAL"""
        sentence = self.sentence
        token = sentence[self.head]
        if token.FORM == '':
            return 'LIST'
        elif token.UPOS == 'PRON':
            return 'PRONOMIAL'
        elif token.UPOS == 'PROPN':
            return 'PROPER'

        return 'NOMINAL'

    def nested(self):
        """A mention is nested if any of its parents are a mention."""
        sentence = self.sentence

        # dont hang on (bad) dependency graphs that are not a single tree
        visited = ['0']

        token = sentence[self.head]
        visited.append(token.ID)
        while token.HEAD not in visited:
            token = sentence[token.HEAD]
            visited.append(token.ID)
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
    while idx < len(sentA):
        if sentA[idx].DEPREL == 'flat':
            headA_contents.append(sentA[idx].FORM.lower())
            idx = idx + 1
        else:
            break

    # build the multi word string for mention B
    headB_contents = []
    sentB = mentionB.sentence

    headB_contents.append(sentB[mentionB.head].FORM.lower())
    idx = sentB.index(mentionB.head) + 1
    while idx < len(sentB):
        if sentB[idx].DEPREL == 'flat':
            headB_contents.append(sentB[idx].FORM.lower())
            idx = idx + 1
        else:
            break

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
    # TODO: token.UPOS == 'PRON'?
    sentA = mentionA.sentence
    sentB = mentionB.sentence

    contentA = []
    for id in mentionA.ids:
        token = sentA[id]
        if token.UPOS in ['PROPN', 'NOUN']:
            contentA.append(token.FORM.lower())

    contentB = []
    for id in mentionB.ids:
        token = sentB[id]
        if token.UPOS in ['PROPN', 'NOUN']:
            contentB.append(token.FORM.lower())

    if len(contentA) == 0 or len(contentB) == 0:
        return 0.0

    if len(contentA) <= len(contentB):
        for FORM in contentA:
            if FORM not in contentB:
                return 0.0
    else:
        for FORM in contentB:
            if FORM not in contentA:
                return 0.0

    return 1.0


def mentions_overlap(mentionA, mentionB):
    """
    The mentions are from the same sentence, and overlap in the sentence.
    """
    sentA = mentionA.sentence
    sentB = mentionB.sentence

    if sentA.doc_rank != sentB.doc_rank:
        return 0.0
    if sentA.sent_rank != sentB.sent_rank:
        return 0.0

    if mentionA.head in mentionB.ids or mentionB.head in mentionA.ids:
        return 1.0

    return 0.0


def features_mention(mention):
    # 01_MentionType
    # 02_MentionLength
    # 03_MentionNormLocation
    # 04_IsMentionNested

    sentence = mention.sentence
    dataset = sentence.dataset

    doc_length = list(dataset.doc_lengths.values())[sentence.doc_rank]
    norm_location = sentence.sent_rank / (doc_length - 1.0)

    return torch.cat((
        to_one_hot(mention_type_codec, mention.type()),
        torch.tensor([
            len(mention.ids),
            norm_location,
            mention.nested()
            ])
        ))


def features_mention_pair(mentionA, mentionB):
    # 03_HeadsAgree
    # 04_ExactStringMatch
    # 05_RelaxedStringMatch
    # 06_SentenceDistance
    # 08_Overlapping
    return torch.tensor([
        mentions_heads_agree(mentionA, mentionB),
        mentions_match_exactly(mentionA, mentionB),
        mentions_match_relaxed(mentionA, mentionB),
        abs(mentionA.sentence.sent_rank - mentionB.sentence.sent_rank),
        mentions_overlap(mentionA, mentionB)
        ])


def adjacency_matrix(sentence):
    """
    Build a modified adjacency matrix from the sentence.

    By mulitplying a position vector by the adjacency matrix,
    we do one step along the dependency arcs.
    This is used for constructing the mention text from a head, so not all arcs
    are useful.
    """
    L = np.zeros([len(sentence)]*2, dtype=np.int)
    for token in sentence:
        if token.HEAD == "0" or token.DEPREL in ADJACENCY_SKIP_DEPREL:
            continue
        L[sentence.index(token.ID), sentence.index(token.HEAD)] = 1

    return L


def build_mentions_from_heads(sentence, heads):
    """
    Build the mention contents from the given heads by
    gathering leaves from the subtree.  The subtree is pruned by removing all
    direct descendants with a DEPREL in a black list, MENTION_REMOVE_DESC.
    This list is based on statistics over the SoNaR dataset.

    heads is a list of Token.ID

    Returns a list of Mention
    """
    to_descendants = adjacency_matrix(sentence)

    # Raise the matrix to the len(sentence) power; this covers the
    # case where the tree has maximum depth. But first add the unity
    # matrix, so the starting word, and all words we reach, keep a non-zero
    # value.
    is_descendant = to_descendants + np.eye(len(sentence), dtype=np.int)
    is_descendant = np.linalg.matrix_power(is_descendant, len(sentence))

    # collect the spans
    mentions = []
    for head in heads:
        if sentence[head].DEPREL == 'punct':
            mentions.append(
                Mention(
                    head=head,
                    sentence=sentence,
                    refid=sentence[head].COREF,
                    start=head,
                    end=head,
                    ids=[head]
                    )
            )
            continue

        # get all tokens that make up the subtree, by construction
        ids, = np.where(is_descendant[:, sentence.index(head)] > 0)

        # Prune direct descendants at this point by looking at the deprel to
        # our subtree root.
        # This deals with leading 'en, ook, als, in, op, voor, bij, ...'
        # that are not part of the mention.
        # Then remove [ADV, CCONJ, SCONJ], and
        # then [ADP, ADV] from the front.
        # Also remove empty tokens (could have been added by our preprocesing)
        ids_to_prune = []
        for token in sentence:
            if token.HEAD == head and token.DEPREL in MENTION_REMOVE_DESC:
                prune, = np.where(
                        is_descendant[:, sentence.index(token.ID)] > 0
                        )
                ids_to_prune += list(prune)

        pruned_ids = []
        for id in ids:
            if id not in ids_to_prune:
                pruned_ids.append(id)

        if len(pruned_ids) == 0:
            # malformed dependency tree
            logging.error('Issue with token {}, sentence: {} {}'.format(
                head, sentence.doc_id, sentence.sent_id)
                )
            continue

        # removing leading tokens
        if len(pruned_ids) > 1 and \
                sentence[pruned_ids[0]].UPOS in MENTION_REMOVE_LEADING:
            pruned_ids = pruned_ids[1:]

        # we have a potential issue with the extra, empty, list tokens
        # (they are added by transform_coordinations)
        # When validating, empty tokens are not printed, and the span
        # would be malformed.
        # move the end of the span until it is not an empty token.
        id_start = pruned_ids[0]
        id_end = pruned_ids[-1]
        while sentence[id_end].FORM == '' and id_end > id_start:
            id_end -= 1

        if len(pruned_ids) > 0:
            mentions.append(
                Mention(
                    head=head,
                    sentence=sentence,
                    refid=sentence[head].COREF,
                    start=sentence[id_start].ID,
                    end=sentence[id_end].ID,
                    ids=[sentence[i].ID for i in pruned_ids]
                    )
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


def mark_anaphoric_mentions(dataset):
    """
    Set the Token.AN for each mention in the dataset.
    """
    doc_rank = 0
    entities = {}
    for sentence in dataset:
        if sentence.doc_rank != doc_rank:
            doc_rank = sentence.doc_rank
            entities = {}
        for token in sentence:
            if token.COREF == '_':
                continue
            if token.COREF in entities:
                entities[token.COREF] += 1
                token.AN = 1.0
            else:
                entities[token.COREF] = 1
                token.AN = 0.0
    return


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

        sentence.add(tokenX)

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


def get_mentions_from_bra_ket(sentence):
    """
    Get mentions from the sentence' brak-ket coref annotation.
    """
    mentions = []
    for token in sentence:
        refs = token.COREF.split('|')
        for ref in refs:
            ms = ref_start.match(ref)
            me = ref_end.match(ref)
            mo = ref_one.match(ref)

            if ref == '_':
                # not a mention
                continue

            # start of a mention: create a new span
            if ms:
                refid, = ms.groups(0)
                mentions.append(Mention(
                    sentence=sentence,
                    refid=refid,
                    start=token.ID,
                    end='_'
                    )
                )

            # end of a mention: close a span
            elif me:
                refid, = me.groups(0)
                menid = len(mentions) - 1
                # find the most recent mention with this refid,
                # ie. top-of-the-stack
                while menid >= 0 and (
                        mentions[menid].refid != refid or
                        mentions[menid].end != '_'
                        ):
                    menid -= 1

                mentions[menid].end = token.ID

            # single word mention
            elif mo:
                refid, = mo.groups(0)
                mentions.append(Mention(
                    sentence=sentence,
                    refid=refid,
                    start=token.ID,
                    end=token.ID
                    )
                )

        # gather the tokens that make up the span
        if token.DEPREL != 'punct':
            for mention in mentions:
                if mention.end in ['_', token.ID]:
                    mention.ids.append(token.ID)

    # sanity check: are all spans closed?
    for mention in mentions:
        if mention.end == '_':
            logging.error('Unclosed mention found {} {} {}'.format(
                sentence.doc_id, sentence.sent_id, mention.refid)
                )
            mention.end = sentence[-1].ID

    return mentions


def most_similar_mention(target, candidates):
    """
    Find the head-based mention most similar to the given span,
    based on the simple matching coefficient.

        target:     the Mention to approximate
        candidates: a list of Mention

    Returns:

        Mention
    """
    # shortcuts: corner cases
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return candidates[0]

    # score each candidate
    best = -1.0
    mention = None
    for candidate in candidates:
        current = 0
        if candidate.sentence == target.sentence:
            for token in target.sentence:
                A = token.ID in target.ids
                B = token.ID in candidate.ids
                if A == B or token.FORM == '':
                    current += 1

        else:
            # mentions from different sentence
            current = -1

        if current == len(target.sentence):
            # shortcut: maximum score
            return candidate

        elif current > best:
            mention = candidate
            best = current

    return mention


def convert_mentions(mentions):
    """
    Convert bra-ket mentions to head based mentions.

    For each mention, look at its ids and find a syntactic head that
    matches the span best. A head can only match a single mention, so
    the conversion will not always work. Only succesful conversions are
    returned.

        mentions   list of Mention (bra-ket)

    Returns:
        a list of Mention (head based)

    NOTE: the mentions are new objects, the original mentions are unchanged.
    """
    # shortcut
    if len(mentions) == 0:
        return []

    result = []

    # group the mentions by sentence
    mentions_per_sentence = {}
    for mention in mentions:
        sentence = mention.sentence
        if sentence in mentions_per_sentence:
            mentions_per_sentence[sentence].append(mention)
        else:
            mentions_per_sentence[sentence] = [mention]

    # process sentence by sentence
    for sentence in mentions_per_sentence.keys():
        # the candidate spans:
        # all possible subtrees except punctuation
        heads_to_try = []
        for token in sentence:
            if token.DEPREL != 'punct':
                heads_to_try.append(token.ID)
        candidates = build_mentions_from_heads(mention.sentence, heads_to_try)

        # assign the best candidate for each mention
        assigned_candidates = []
        for mention in mentions_per_sentence[sentence]:
            best_mention = most_similar_mention(mention, candidates)

            if best_mention is None:
                # headless mention (only punctuation)
                assigned_candidates.append(None)
            elif best_mention in assigned_candidates:
                # Two mentions claim the same head; conceptually something like
                # 1 John      ....  ..     (0|(0)
                # 2 Johnson   flat   1     0)
                idx = assigned_candidates.index(best_mention)
                competing_mention = mentions[idx]

                if len(mention.ids) > len(competing_mention.ids):
                    # take the longer span, as it carries more information
                    assigned_candidates[idx] = None
                    assigned_candidates.append(best_mention)
                else:
                    # for similar length, take the one first encountered
                    # so drop this mention
                    assigned_candidates.append(None)
            else:
                # the first span to claim this mention
                assigned_candidates.append(best_mention)

        converted_mentions = []
        for mention, candidate in \
                zip(mentions_per_sentence[sentence], assigned_candidates):
            if candidate is not None:
                candidate.refid = mention.refid
                converted_mentions.append(candidate)

        result += converted_mentions

    return result


def preprocess_sentence(sentence):
    """
    Preprocess a Sentence containing coref annotation in bra-ket notation.

    Modify the coordination and copula in the dependency tree
    Replace span based mentions with syntactic head based mentions

    Returns:
        list of Mention (bra-ket), list of Mention (head based)
    """
    sentence = transform_tree(sentence)
    bra_ket_mentions = get_mentions_from_bra_ket(sentence)
    head_mentions = convert_mentions(bra_ket_mentions)

    # clear bra-ket annotations
    for token in sentence:
        token.COREF = '_'

    # add head based annotations
    for mention in head_mentions:
        sentence[mention.head].COREF = mention.refid

    return bra_ket_mentions, head_mentions


def postprocess_sentence(sentence):
    """
    Postprocess a Sentence containing syntactic head based mentions.

    Replace the syntactic head based mentions by spans.

    NOTE: Does *NOT* undo changes made to the dependency graph.
    """
    # get head based mentions
    mentions = get_mentions(sentence)

    # clear head based annotation
    for token in sentence:
        token.COREF = '_'

    # add bra-ket annotation
    for mention in mentions:
        astart = sentence[mention.start].COREF
        aend = sentence[mention.end].COREF

        if mention.start == mention.end:
            # add a single token mention
            if astart == '_':
                astart = '({})'.format(mention.refid)
            else:
                astart = '{}|({})'.format(astart, mention.refid)
        else:
            # add a multi token mention
            if astart == '_':
                astart = '({}'.format(mention.refid)
            else:
                astart = '({}|{}'.format(mention.refid, astart)

            if aend == '_':
                aend = '{})'.format(mention.refid)
            else:
                aend = '{})|{}'.format(mention.refid, aend)

            sentence[mention.end].COREF = aend
        sentence[mention.start].COREF = astart

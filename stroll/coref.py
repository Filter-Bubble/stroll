import re

import numpy as np

import torch
import dgl

import logging

from stroll.conllu import transform_tree
from stroll.labels import to_one_hot, mention_type_codec
from stroll.labels import feats_codec, deprel_codec

from functools import lru_cache

MAX_MENTION_DISTANCE = 50

# We always ignore punctuation, as they have the sentence root as head
# For multiple heads per span, we take the 1st head (ie. with lowest token.ID)
# Spans can be headless at this point; these are always spans conataining
# only punctuation. Drop them.
# There can be mulitple spans with the same head, take the first occuring.
# Trim the span by removing specific UPOS tokens from the front (ADP, CCONJ,..)

ref_start = re.compile('^\((\d+)$')
ref_end = re.compile('^(\d+)\)$')
ref_one = re.compile('^\((\d+)\)$')

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
      ids       List of Token.ID of all tokens part of the mention
      anaphore  If the mention is an anaphore (1.0) or not (0.0)
      mwt       The set of token.FORM.lower() of a possible multi word token
      mods      The set of token.FORM.lower() of descendant amod and nmod

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
                 ids=None,  # The IDs of all tokens part of the mention,
                 anaphore=1.0  # If the mention is an anaphore
                 ):
        self.sentence = sentence
        self.head = head
        self.refid = refid
        self.start = start
        self.end = end
        self.anaphore = anaphore
        if ids:
            self.ids = ids
        else:
            self.ids = []
        if self.head and self.sentence:
            self.mwt = get_multi_word_token(sentence, self.head)
            self.modifiers = get_modifiers_for_token(sentence, self.head)
        else:
            self.mwt = None
            self.modifiers = None

    def __repr__(self):
        p = ""
        p += '# sent_id:   {}\n'.format(self.sentence.sent_id)
        p += '# full_text: {}\n'.format(self.sentence.full_text)
        p += 'refid= {}\n'.format(self.refid)
        if self.head:
            p += 'head=  {} {}\n'.format(self.head, self.sentence[self.head].FORM)
        p += 'span=  {}-{}\n'.format(self.start, self.end)
        p += 'text= {}\n'.format([self.sentence[i].FORM for i in self.ids])
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
        elif token.UPOS == 'X':
            # 'Foreign=Yes'
            # NOTE: We cant really separate foreign language quotes and nouns
            # from proper nouns.
            # If we treat them as nouns, I dont expect the word vectors to
            # be useful. If we assume they are proper nouns, we'll be doing
            # exact string matches which make sense even for foreign words.

            # 'Abbr=Yes'
            # NOTE: This token is the head of a mention, and abbreviations are
            # probably proper nouns. 'etc.' and 'o.a.' would not be the head.
            return 'PROPER'

        return 'NOMINAL'

    def get_identifier(self):
        sentence = self.sentence
        return '{}_{}_{}'.format(
                sentence.doc_id,
                sentence.sent_id,
                self.head)

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


def get_multi_word_token(sentence, start_id):
    """
    Return the set of all token.FORM.lower() that are directly descendant
    from the start token.
    """
    start = sentence[start_id]
    mwt = set([clean_token(start.FORM)])

    for token in sentence:
        if token.HEAD == start.ID and token.DEPREL in \
                ['flat', 'fixed', 'compound:prt']:
            mwt.add(clean_token(token.FORM))
    return mwt


def get_modifiers_for_token(sentence, start_id):
    """
    Return the set of modifiers (token.FORM.lower()) that directly attach
    to the start token and have DEPREL 'nmod' or 'amod'.
    """
    start = sentence[start_id]
    mods = set()

    for token in sentence:
        if token.HEAD == start.ID and token.DEPREL in ['nmod', 'amod']:
            mods.add(clean_token(token.FORM))
    return mods


def mentions_heads_agree(mentionA, mentionB):
    """The (uncased) head.FORM of the mentions match"""

    # All tokens from the shorter one should be in the larger
    if mentionA.mwt.issubset(mentionB.mwt) or \
       mentionB.mwt.issubset(mentionA.mwt):
        return 1.0
    else:
        return 0.0


def mentions_match_exactly(mentionA, mentionB):
    """The (uncased) sentence[ids].FORM of the mentions match"""
    if len(mentionA.ids) != len(mentionB.ids):
        return 0.0

    sentA = mentionA.sentence
    sentB = mentionB.sentence
    for i in range(len(mentionA.ids)):
        formA = clean_token(sentA[mentionA.ids[i]].FORM)
        formB = clean_token(sentB[mentionB.ids[i]].FORM)
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
            contentA.append(clean_token(token.FORM))

    contentB = []
    for id in mentionB.ids:
        token = sentB[id]
        if token.UPOS in ['PROPN', 'NOUN']:
            contentB.append(clean_token(token.FORM))

    if len(contentA) == 0 or len(contentB) == 0:
        return 0.0

    if len(contentA) <= len(contentB):
        for FORM in contentA:
            if FORM not in contentB:
                return 0
        return len(contentA) / len(contentB)
    else:
        for FORM in contentB:
            if FORM not in contentA:
                return 0.0
        return len(contentB) / len(contentA)


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


@lru_cache(maxsize=500)
def clean_token(text):
    return re.sub('[^\w]+', '', text).lower()


@lru_cache(maxsize=500)
def features_mention(mention):
    # 01_MentionType                 4
    #    FEATS                       33
    #    DEPREL                      36
    # 02_MentionLength               1
    # 03_MentionNormLocation         1
    # 04_IsMentionNested             1
    # total                          76

    sentence = mention.sentence
    dataset = sentence.dataset

    doc_length = list(dataset.doc_lengths.values())[sentence.doc_rank]
    norm_location = sentence.sent_rank / (doc_length - 1.0)

    return torch.cat((
        to_one_hot(mention_type_codec, mention.type()),
        to_one_hot(feats_codec, sentence[mention.head].FEATS.split('|')),
        to_one_hot(deprel_codec, sentence[mention.head].DEPREL),
        torch.tensor([
            len(mention.ids),
            norm_location,
            mention.nested()
            ])
        ))


@lru_cache(maxsize=500)
def features_mention_pair(mentionA, mentionB):
    # 03_HeadsAgree             1
    # 04_ExactStringMatch       1
    # 05_RelaxedStringMatch     1
    # 06_SentenceDistance       1
    # 08_Overlapping            1
    #                           5
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
    mentions = {}
    for head in heads:
        if sentence[head].DEPREL == 'punct':
            mentions[(head, head)] = Mention(
                    head=head,
                    sentence=sentence,
                    refid=sentence[head].COREF,
                    start=head,
                    end=head,
                    ids=[head],
                    anaphore=sentence[head].anaphore
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
            if (id not in ids_to_prune) and (sentence[id].FORM != ''):
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

        id_start = pruned_ids[0]
        id_end = pruned_ids[-1]

        if len(pruned_ids) > 0:
            start = sentence[id_start].ID
            end = sentence[id_end].ID
            if (start, end) in mentions and len(pruned_ids) < len(mentions[(start, end)].ids):
                continue
            else:
                mentions[(start, end)] = Mention(
                        head=head,
                        sentence=sentence,
                        refid=sentence[head].COREF,
                        start=sentence[id_start].ID,
                        end=sentence[id_end].ID,
                        ids=[sentence[i].ID for i in pruned_ids],
                        anaphore=sentence[head].anaphore
                        )

    # TODO: how to correctly prune non consecutive / double mentions?
    return list(mentions.values())


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


def mark_gold_anaphores(dataset):
    """
    Set the Token.anaphore for each mention in the dataset from the current
    head-based token.COREF clusters.
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
                token.anaphore = 1.0
            else:
                entities[token.COREF] = 1
                token.anaphore = 0.0


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
    spans = [(m.start, m.end) for m in mentions]
    if len(spans) > len(set(spans)):
        logging.warning('Double mention spans in sentence: "{}"'.format(sentence))
        breakpoint()
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


def nearest_linking(similarities, anaphores, margin=0):
    nmentions = len(similarities)
    entities = []

    # start a new entity for the first mention
    entities.append(set([0]))

    for i in range(1, nmentions):
        # link if allowed / allowed
        linked = False
        if anaphores[i] > 0.5 + margin:
            # take the similarity to all possible antecendents
            antecedent_sims = similarities[0:i, i]

            # find the most similar
            antecedent = torch.argmax(antecedent_sims)
            if isinstance(antecedent, torch.Tensor):
                # we store plain integers in the sets
                antecedent = antecedent.item()

            # find the set that contains the antecedent
            for entity in entities:
                if antecedent in entity:
                    entity.add(i)
                    linked = True
                    break

        if not linked:
            # start a new entity
            entities.append(set([i]))

    clusters = np.zeros(nmentions)
    for refid, entity in enumerate(entities):
        for i in entity:
            clusters[i] = refid

    return clusters


def mentions_can_link(aid, mid, antecedent, mention):
    """
    Deterimine if mentions are allowed to link:
    they should be from the same document, and withn MAX_MENTION_DISTANCE from
    eachother.
    """
    if mid - aid >= MAX_MENTION_DISTANCE:
        # TODO: fill with exponentially decaying similarity?
        return False

    if antecedent.sentence.doc_rank != mention.sentence.doc_rank:
        # TODO: fill with very low similarities?
        return False

    return True


def predict_similarities(net, mentions, gvec):
    """
        net       a CorefNet instance
        mentions  a list of Mentions
        gvec      the graph-convolutioned vectors for the mentions

    returns:
      similarities   torch.tensor(nmentions, nmentions)
      link           torch.tensor(nmentions, nmentions)
    """

    nmentions = len(mentions)
    links = torch.zeros([nmentions, nmentions])
    similarities = torch.ones([nmentions, nmentions]) * -1e8

    # build a list of antecedents, and the pair vectors
    vectors = []
    aids, mids = np.triu_indices(nmentions, 1)
    for aid, mid in zip(aids, mids):
        if not mentions_can_link(aid, mid,
                                 mentions[aid], mentions[mid]):
            continue

        antecedent = mentions[aid]
        mention = mentions[mid]

        if antecedent.refid == mention.refid:
            links[aid, mid] = 1
            links[mid, aid] = 1

        # build pair (aidx, midx)
        vectors.append(
            torch.cat((
                gvec[mid].view(-1),
                features_mention(mention),
                features_mention_pair(
                    antecedent,
                    mention),
                gvec[aid].view(-1),
                features_mention(antecedent)
                )
            )
        )

    # get the similarity between those pairs
    pairsim = net.task_b(torch.stack(vectors))

    p = 0
    for aid, mid in zip(aids, mids):
        if not mentions_can_link(aid, mid,
                                 mentions[aid], mentions[mid]):
            continue

        similarities[aid, mid] = pairsim[p]
        similarities[mid, aid] = similarities[aid, mid]
        p += 1

    return links, similarities


def predict_anaphores(net, mentions):
    """
        net       a CorefNet instance
        mentions  a list of Mentions

    returns:
      anpahoric torch.tensor(nmentions )
    """
    nmentions = len(mentions)
    anaphore = []

    for mid in range(nmentions):
        mention = mentions[mid]
        vectors = []
        for aid in range(0, mid):
            if not mentions_can_link(aid, mid,
                                     mentions[aid], mentions[mid]):
                continue

            antecedent = mentions[aid]

            # build pair (aidx, midx)
            vectors.append(torch.cat((
                    features_mention(mention),
                    features_mention_pair(
                        antecedent,
                        mention),
            )))

        if len(vectors) > 0:
            anaphore.append(torch.mean(
                torch.stack(vectors),
                dim=0
            ))
        else:
            # no antecedents, so cannot be an anaphore
            anaphore.append(torch.zeros(80))

    # feed the vectors through the network, and return the result
    return net.task_a(torch.stack(anaphore))


def coref_collate(batch):
    """
    Collate function to batch samples together.
    """

    mentions = []
    for g in batch:
        mentions += get_mentions(g.sentence)
    return dgl.batch(batch), mentions

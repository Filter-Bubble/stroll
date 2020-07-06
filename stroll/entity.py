from functools import lru_cache

import torch
import numpy as np

from stroll.labels import to_one_hot
from stroll.labels import feats_codec, deprel_codec, mention_type_codec

MAX_CANDIDATES = 20

wordvector = None


class Entity():
    def __init__(self):
        self.mentions = []
        self.gold_entities = {}
        self.proper_nouns = set()
        self.nouns = set()
        self.pronouns = set()
        self.modifiers = set()
        self.features = torch.zeros(len(feats_codec.classes_))

    def __repr__(self):
        types = {
                'LIST': 0,
                'PRONOMIAL': 0,
                'PROPER': 0,
                'NOMINAL': 0
                }
        for mention in self.mentions:
            types[mention.type()] += 1

        res = ''
        res += 'Length       {:4d}\n'.format(len(self.mentions))

        res += 'Nouns        {:4d}:'.format(types['NOMINAL'])
        res += ' '.join(list(self.nouns)) + '\n'

        res += 'Proper Nouns {:4d}:'.format(types['PROPER'])
        res += ' '.join(list(self.proper_nouns)) + '\n'

        res += 'Pronouns     {:4d}:'.format(types['PRONOMIAL'])
        res += ' '.join(list(self.pronouns)) + '\n'

        res += '\n'
        res += 'Modifiers   :' + ' '.join(list(self.modifiers)) + '\n'
        res += 'Features    :' + list(self.features.numpy()).__repr__() + '\n'
        return res

    def _calculate_features(self):
        features = []
        for mention in self.mentions:
            features.append(features_for_mention(mention))
        self.features = torch.mean(torch.stack(features), dim=0)
        self.features.detach()

    def quantized_length(self):
        if len(self.mentions) == 1:
            ql = torch.tensor([0.0])
        elif len(self.mentions) < 4:
            ql = torch.tensor([0.5])
        else:
            ql = torch.tensor([1.0])
        return ql

    def add(self, mention):
        self.mentions.append(mention)
        gold_entity = mention.refid

        # keep track of the (multiword) tokens for exact string matches
        if mention.type() == 'PROPER':
            # build the (multi word) string for the proper noun
            sentence = mention.sentence

            # add the head
            self.proper_nouns.add(sentence[mention.head].FORM.lower())

            # add all following tokens with DEPREL == 'flat'
            idx = sentence.index(mention.head) + 1
            while idx < len(sentence):
                if sentence[idx].DEPREL == 'flat':
                    self.proper_nouns.add(sentence[idx].FORM.lower())
                    idx = idx + 1
                else:
                    break
        elif mention.type() == 'NOMINAL':
            sentence = mention.sentence
            self.nouns.add(sentence[mention.head].FORM.lower())

        elif mention.type() == 'PRONOMIAL':
            sentence = mention.sentence
            self.pronouns.add(sentence[mention.head].FORM.lower())

        # the gold annotation
        if gold_entity in self.gold_entities:
            self.gold_entities[gold_entity] += 1
        else:
            self.gold_entities[gold_entity] = 1

        # precaluate features from the conll FEATS column
        self._calculate_features()

        # modifiers:
        # add all 'amon' and 'nmod' directly attached to the mentions' head
        self.modifiers = self.modifiers.union(modifiers_for_mention(mention))

    def as_set(self):
        return set(
                [mention.get_identifier() for mention in self.mentions]
                )


def set_wordvector(model):
    global wordvector

    wordvector = model


@lru_cache(maxsize=100)
def modifiers_for_mention(mention):
    """
    Return the set of modifiers (token.FORM.lower()) that directly attach
    to the mentions' head (ie. have DEPREL 'nmod' or 'amod').
    """
    mods = set()

    sentence = mention.sentence

    for id in mention.ids:
        token = sentence[id]
        if token.HEAD == mention.head:
            # attach directly to the mentions' head
            if token.DEPREL in ['nmod', 'amod']:
                mods.add(token.FORM.lower())
    return mods


def modifier_agreement_entity_mention(entity, mention):
    """
    The ratio of shared modifiers between the mention and the entity
    to the number of modifiers of the mention.
    dims = 1
    """
    mods = modifiers_for_mention(mention)

    if len(mods) == 0 or len(entity.modifiers) == 0:
        return torch.zeros(1)

    shared_mods = len(entity.modifiers.intersection(mods))

    return torch.tensor([1.0 * shared_mods / len(mods)])


def quantized_distance_top_to_mention(entity, mention):
    """
    The distance between the most recent mention of the entity,
    and the query mention.
    dim = 1
    """
    # Distance form the entity
    # assume the mentions are from the same document
    top = entity.mentions[-1]
    distance = abs(mention.sentence.sent_rank - top.sentence.sent_rank)
    if distance > 3:
        return torch.tensor([0.0])
    else:
        return torch.tensor([(3.0 - distance)/3])


def semantic_role_similiarty_top_to_mention(entity, mention):
    """
    Dot product between DEPREL vectors of the most recent mention the entity,
    and the query mention.
    dim = 1
    """
    # Similar semantic role
    top = entity.mentions[-1]
    key = to_one_hot(deprel_codec, top.sentence[top.head].DEPREL)
    query = to_one_hot(deprel_codec, mention.sentence[mention.head].DEPREL)

    return torch.sum(query * key, dim=0, keepdim=True)


def wordvector_similarity_entity_to_mention(entity, mention):
    """
    Median of the dot product between the normalized wordvector of the mention,
    and each mention in the entity.

    Median value for two random vectors is 0.24

    dim = 1
    """
    dots = []
    sentence = mention.sentence
    query = wordvector[sentence[mention.head].FORM.lower()]

    def comp_wv(wa, wb):
        """
        Dot product between normalized vectors
        """
        wan = wa / (np.dot(wa, wa)**0.5)
        wbn = wb / (np.dot(wb, wb)**0.5)
        return np.dot(wan, wbn)

    if mention.type() == 'NOMINAL' and len(entity.nouns) > 0:
        for key in entity.nouns:
            dots.append(comp_wv(query, wordvector[key]))
    elif mention.type() == 'PROPER' and len(entity.proper_nouns) > 0:
        for key in entity.proper_nouns:
            dots.append(comp_wv(query, wordvector[key]))
    else:
        # LIST PRONOMIAL
        return torch.tensor([0.24, 0.24])

    median = np.median(dots)
    best = np.max(dots)
    return torch.tensor([median, best])


def string_match_entity_to_mention(entity, mention):
    """
    If the head of the mention is contained in the entity; for proper nouns
    the fraction of tokens in the multi word token.
    dim = 1
    """
    matching_words = 0.0

    sentence = mention.sentence
    if mention.type() == 'NOMINAL':
        if sentence[mention.head].FORM.lower() in entity.nouns:
            matching_words = 1.0
    elif mention.type() == 'PROPER':
        # the head
        multiword_length = 1
        if sentence[mention.head].FORM.lower() in entity.proper_nouns:
            matching_words = 1.0

        # look at all following tokens with DEPREL == 'flat'
        idx = sentence.index(mention.head) + 1
        while idx < len(sentence):
            if sentence[idx].DEPREL == 'flat':
                multiword_length += 1
                if sentence[idx].FORM.lower() in entity.proper_nouns:
                    matching_words += 1
            else:
                break
            idx += 1
        matching_words = matching_words / multiword_length

    return torch.tensor([matching_words])


def precise_construct_mention_mention(ma, mb):
    """
    Some features from section 3.3.4 of:
    Deterministic Coreference Resolution Based on Entity-Centric,
    Precision-Ranked Rules

      appositive: The chairman of the company, mr. X, ...
                  mentions separated by punctuation,
                  but not in conj to the same head

      predicate nominative: A is B
                  in our transformed dependency graph:
                  the mentions share the same head
                  one of the mentions has DEPREL 'cop'

    And a custom rule:

      same conjunction: [[A] and [B]]

    dim : 3
    """
    # I did not implement those:
    # Acronym     Three Letter Acronym TLA
    #             all capital letters in the multiword token
    #             -> check for a few acronyms, but the annotation lists
    #             them as a single entity: [Three Letter Acronym (TLA)]
    # Role Appositive    [[Actress] Jane Doe]
    #                    same issue as for acronyms
    # Demonym            need lists for Dutch
    # Relative pronoun   [the finance street [which] has already formed
    #                    in the Waitan district])
    if mb.sentence != ma.sentence:
        return torch.zeros(3)

    sentence = mb.sentence

    adr = sentence[ma.head].DEPREL
    bdr = sentence[mb.head].DEPREL

    ap = 0.0  # appositive
    pn = 0.0  # predicate nominative
    sc = 0.0  # part of same conjunction

    if ma.head == mb.head:
        if adr == 'cop' or bdr == 'cop':
            # predicate nominative
            pn = 1.0
        else:
            pn = 0.0
        if bdr == 'conj' and adr == 'conj':
            # both members of the same conjunction [A], [B], ....
            sc = 1.0

    if (ma.head == sentence[mb.head].ID and adr == 'conj') or \
       (mb.head == sentence[ma.head].ID and bdr == 'conj'):
        # whole list and a member of the list
        sc = 1.0

    # not both part of a list (conjunction)
    if not (adr == 'conj' and bdr == 'conj'):
        if mb.ids[-1] < ma.ids[0]:
            # the order is: mb - ma
            separation = sentence.index(ma.ids[0]) - \
                    sentence.index(mb.ids[-1])
            separator = sentence.index(mb.ids[-1]) + 1
        else:
            # the order is: ma - mb
            separation = sentence.index(mb.ids[0]) - \
                    sentence.index(ma.ids[-1])
            separator = sentence.index(ma.ids[-1]) + 1

        if separation == 1 or (  # BUGFIX after run 32
                separation == 2 and sentence[separator].DEPREL == 'punct'
                ):
            # directly adjacent or separated by a punctuation
            ap = 1.0

    return torch.tensor([ap, pn, sc])


def precise_constructs_top_to_mention(entity, mention):
    top = entity.mentions[-1]
    return precise_construct_mention_mention(top, mention)


def features_for_mention(mention):
    """
    Features from the conll FEATS for the mention.
    len(feats_codec.classes_) == 32
    """
    sentence = mention.sentence
    return to_one_hot(feats_codec, sentence[mention.head].FEATS.split('|'))


def action_new_probability(net, entities, mention):
    """
    Probabilities to start a new entity.
    """
    sentence = mention.sentence
    dataset = sentence.dataset

    # Relative position in document : 1
    doc_length = list(dataset.doc_lengths.values())[sentence.doc_rank]
    norm_location = sentence.sent_rank / (doc_length - 1.0)

    # Ratio of mentions to entities : 1
    total_mentions = 0
    for entity in entities:
        total_mentions += len(entity.mentions)
    mentions_to_entities_ratio = total_mentions / (len(entities) + 0.001)

    # Number of candidates : 1
    ncandidates = min(len(entities), MAX_CANDIDATES) * 1.0

    # Mention type : 4
    mtype = to_one_hot(mention_type_codec, mention.type())

    # Length of the mention : 1
    mlength = len(mention.ids)

    query = torch.cat([
        torch.tensor([
            mentions_to_entities_ratio,
            norm_location,
            ncandidates,
            mlength]),
        mtype
        ])

    return net.new_entity_prob(query)


def action_add_probabilities(net, entities=[], mention=None):
    """
    Probabilities to add the mention to an existing entity.
    """

    # build matrix with evidence:
    input = []
    for entity in entities:
        input.append(torch.cat([
            #  * 1 quantized entity size
            entity.quantized_length(),

            #  * 2 wordvector_similarity_entity_to_mention
            wordvector_similarity_entity_to_mention(entity, mention),

            #  * 1 string_match_entity_to_mention
            string_match_entity_to_mention(entity, mention),

            #  * 1 modifiers agreement
            modifier_agreement_entity_mention(entity, mention),

            #  * 32 feature match entity
            entity.features * features_for_mention(mention),

            #  * 4 mention type
            to_one_hot(mention_type_codec, mention.type()),

            #  * 1 semantic_role_similiarty_top_to_mention
            semantic_role_similiarty_top_to_mention(entity, mention),

            #  * 1 quantized_distance_top_to_mention
            quantized_distance_top_to_mention(entity, mention),

            # * 3 precise_constructs_top_to_mention
            precise_constructs_top_to_mention(entity, mention)
            ]))

    while len(input) < MAX_CANDIDATES:
        input.append(torch.zeros(46))

    # pass through network MAX_CANDIDATES * 51 -> MAX_CANDIDATES
    input = torch.cat(input)
    return net.combine_evidence(input)

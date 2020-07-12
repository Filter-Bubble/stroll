import torch
import numpy as np

from stroll.labels import to_one_hot
from stroll.labels import mention_type_codec
from stroll.coref import get_multi_word_token
from stroll.coref import clean_token

MAX_CANDIDATES = 30

wordvector = None


class Entity():
    def __init__(self):
        self.mentions = []
        self.refid = 0
        self.gold_entities = {}
        self.proper_nouns = set()
        self.nouns = set()
        self.pronouns = set()
        self.modifiers = set()
        self.features = torch.zeros(5)

    def __repr__(self):
        types = {
                'LIST': 0,
                'PRONOMIAL': 0,
                'PROPER': 0,
                'NOMINAL': 0
                }
        for mention in self.mentions:
            types[mention.type()] += 1

        res = 'Refid: {}'.format(self.refid)
        res += 'Length       {:4d}\n'.format(len(self.mentions))

        res += 'Nouns        {:4d}: '.format(types['NOMINAL'])
        res += ' '.join(list(self.nouns)) + '\n'

        res += 'Proper Nouns {:4d}: '.format(types['PROPER'])
        res += ' '.join(list(self.proper_nouns)) + '\n'

        res += 'Pronouns     {:4d}: '.format(types['PRONOMIAL'])
        res += ' '.join(list(self.pronouns)) + '\n'

        res += '\n'
        res += 'Modifiers   : ' + ' '.join(list(self.modifiers)) + '\n'
        res += 'Features    : ' + list(self.features.numpy()).__repr__() + '\n'
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
            # add the (multi word) string for the proper noun
            self.proper_nouns = self.proper_nouns.union(mention.mwt)

        elif mention.type() == 'NOMINAL':
            sentence = mention.sentence
            self.nouns.add(clean_token(sentence[mention.head].LEMMA))

            # add role appositive proper nouns 'architect de Sola'
            for token in sentence:
                if token.HEAD == mention.head and \
                        token.DEPREL == 'appos' and \
                        token.UPOS in ['PROPN', 'X']:
                    self.proper_nouns = self.proper_nouns.union(
                        get_multi_word_token(sentence, token.ID)
                        )

        elif mention.type() == 'PRONOMIAL':
            sentence = mention.sentence
            self.pronouns.add(clean_token(sentence[mention.head].FORM))

        # the gold annotation
        if gold_entity in self.gold_entities:
            self.gold_entities[gold_entity] += 1
        else:
            self.gold_entities[gold_entity] = 1

        # precaluate features from the conll FEATS column
        self._calculate_features()

        # modifiers:
        self.modifiers = self.modifiers.union(mention.modifiers)

    def as_set(self):
        return set(
                [mention.get_identifier() for mention in self.mentions]
                )


def set_wordvector(model):
    global wordvector

    wordvector = model


def modifier_agreement_entity_mention(entity, mention):
    """
    The ratio of shared modifiers between the mention and the entity
    to the number of modifiers of the mention.
    dims = 1
    """
    if len(mention.modifiers) == 0 or len(entity.modifiers) == 0:
        return torch.zeros(1)

    shared_mods = len(
            entity.modifiers.intersection(
                mention.modifiers
            ))

    return torch.tensor([1.0 * shared_mods / len(mention.modifiers)])


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


def semantic_role_mention(mention):
    """
    (part of the) DEPREL vector of the top mention

    Relation includes:
    nsubj, nsubj:pass, obj, iobj, obl, obl:agent

    dim = 6
    """

    RELS = {
            'nsubj': torch.tensor([0.5, 0., 0., 0., 0., 0.]),
            'nsubj:pass': torch.tensor([0., 0.5, 0., 0., 0., 0.]),
            'obj': torch.tensor([0., 0., 0.5, 0., 0., 0.]),
            'iobj': torch.tensor([0., 0., 0., 0.5, 0., 0.]),
            'obl': torch.tensor([0., 0., 0., 0., 0.5, 0.]),
            'obl:agent': torch.tensor([0., 0., 0., 0., 0., 0.5])
            }

    mdr = mention.sentence[mention.head].DEPREL
    if mdr in RELS:
        mdr = RELS[mdr]
    else:
        mdr = torch.zeros(6)

    return mdr


def comp_wv(wa, wb):
    """
    Dot product between normalized vectors
    """
    wan = wa / (np.dot(wa, wa)**0.5 + 1e-8)
    wbn = wb / (np.dot(wb, wb)**0.5 + 1e-8)
    return np.dot(wan, wbn)


def wordvector_similarity_entity_to_mention(entity, mention):
    """
    Median of the dot product between the normalized wordvector of the mention,
    and each mention in the entity.

    Median value for two random vectors is 0.24

    dim = 1
    """
    dots = []
    sentence = mention.sentence

    if mention.type() == 'NOMINAL' and len(entity.nouns) > 0:
        query = wordvector[clean_token(sentence[mention.head].LEMMA)]
        for key in entity.nouns:
            dots.append(comp_wv(query, wordvector[key]))
    elif mention.type() == 'PROPER' and len(entity.proper_nouns) > 0:
        query = wordvector[clean_token(sentence[mention.head].FORM)]
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
        if clean_token(sentence[mention.head].LEMMA) in entity.nouns:
            matching_words = 1.0
    elif mention.type() == 'PROPER':
        matching_words = len(entity.proper_nouns.intersection(mention.mwt)) \
                / len(mention.mwt)

    return torch.tensor([matching_words])


def precise_constructs_top_to_mention(entity, mention):
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
    # Demonym            need lists for Dutch
    # Relative pronoun   [the finance street [which] has already formed
    #                    in the Waitan district])
    top = entity.mentions[-1]

    ap = 0.0  # appositive
    pn = 0.0  # predicate nominative
    sc = 0.0  # part of same conjunction

    if top.sentence != mention.sentence:
        return torch.tensor([ap, pn, sc])

    sentence = top.sentence

    aid = sentence[top.head].ID
    adr = sentence[top.head].DEPREL
    ahd = sentence[top.head].HEAD

    bid = sentence[mention.head].ID
    bdr = sentence[mention.head].DEPREL
    bhd = sentence[mention.head].HEAD

    if ahd == bhd:
        if adr == 'cop' or bdr == 'cop':
            # predicate nominative
            pn = 1.0
        else:
            pn = 0.0
        if bdr == 'conj' and adr == 'conj':
            # both members of the same conjunction [A], [B], ....
            sc = 1.0

    if (ahd == bid and adr == 'conj') or (bhd == aid and bdr == 'conj'):
        # whole list and a member of the list
        sc = 1.0

    # not both part of a conjunction
    if not (adr == 'conj' and bdr == 'conj'):
        mstart = sentence.index(mention.ids[0])
        mend = sentence.index(mention.ids[-1])

        tstart = sentence.index(top.ids[0])
        tend = sentence.index(top.ids[-1])

        if tend < mstart:
            # the order is: top - mention
            separation = mstart - tend
            separator_id = tend + 1
        elif mend < tstart:
            # the order is: mention - top
            separation = tstart - mend
            separator_id = mend + 1
        else:
            # nested?
            separation = 10

        if separation == 1 or (  # BUGFIX after run 32
                separation == 2 and sentence[separator_id].DEPREL == 'punct'
                ):
            # directly adjacent or separated by a punctuation
            ap = 1.0

    return torch.tensor([ap, pn, sc])


def features_for_mention(mention):
    """
    Features from the conll XPOS for the mention.

    [masculine, feminine, neuter, plural, singular]

    dim : 5

    NOUN : in XPOS   zijd [m/f], onz [n]
           in XPOS   ev [singular], mv [plural]
    PROPN: in XPOS   zijd [m/f], onz [n]
           in XPOS   ev [singular], mv [plural]
    PRON:  in XPOS   masc [m], fem [f], onz [n]
           in XPOS   ev [singular], mv [plural]
    LIST:  -         mv
    """
    sentence = mention.sentence

    feats = set(sentence[mention.head].XPOS.split('|'))
    if 'onz' in feats:
        masculine = 0.0
        feminine = 0.0
        neuter = 1.0
    elif 'masc' in feats:
        masculine = 1.0
        feminine = 0.0
        neuter = 0.0
    elif 'fem' in feats:
        masculine = 0.0
        feminine = 1.0
        neuter = 0.0
    else:  # if 'zijd' in feats:
        neuter = 0.0
        m = comp_wv(wordvector[clean_token(sentence[mention.head].LEMMA)],
                    wordvector['man'])
        f = comp_wv(wordvector[clean_token(sentence[mention.head].LEMMA)],
                    wordvector['vrouw'])
        if m > f:
            masculine = 1.0
            feminine = 0.0
        else:
            masculine = 0.0
            feminine = 1.0

    if 'ev' in feats:
        singular = 1.0
        plural = 0.0
    elif 'mv' in feats:
        singular = 0.0
        plural = 1.0
    elif mention.type() == 'LIST':
        singular = 0.0
        plural = 1.0
    else:
        singular = 0.5
        plural = 0.5  # TODO

    return torch.tensor([masculine, feminine, neuter, singular, plural])


def relative_position_in_sentence(mention):
    sentence = mention.sentence

    # how far at the front of the sentence is the mention : 1
    ff = 1.0 - sentence.index(mention.ids[0]) / (len(sentence) * 1.0)

    # how far to the back of the sentence is the mention : 1
    bb = sentence.index(mention.ids[-1]) / (len(sentence) * 1.0)

    return torch.tensor([ff, bb])


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
            mentions_to_entities_ratio,  # 1
            norm_location,  # 1
            ncandidates,  # 1
            mlength  # 1
            ]),
        semantic_role_mention(mention),  # 6
        mtype,  # 4
        relative_position_in_sentence(mention)  # 2
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

            #  * 5 feature match entity
            entity.features * features_for_mention(mention) * 0.2,

            #  * 4 mention type
            to_one_hot(mention_type_codec, mention.type()),

            #  * 12 semantic_role_similiarty_top_to_mention
            semantic_role_mention(mention),
            semantic_role_mention(entity.mentions[-1]),

            #  * 1 quantized_distance_top_to_mention
            quantized_distance_top_to_mention(entity, mention),

            #  * 3 precise_constructs_top_to_mention
            precise_constructs_top_to_mention(entity, mention),

            #  * 4 relative positions (start / end) in sentence
            relative_position_in_sentence(mention),
            relative_position_in_sentence(entity.mentions[-1])
            ])
        )

    # pass through network 34 -> 1
    return net.combine_evidence(torch.stack(input)).view(-1)

import signal
import argparse
import logging

import fasttext
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler

from scorch.scores import muc, b_cubed, ceaf_e

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence
from stroll.model import EntityNet
from stroll.labels import to_one_hot
from stroll.labels import feats_codec, deprel_codec, mention_type_codec

MAX_CANDIDATES = 25

# Global arguments for dealing with Ctrl-C
global writer
writer = None

global wordvector

parser = argparse.ArgumentParser(
        description='Train an entity centric transition based coreference net'
        )
parser.add_argument(
        '--train',
        dest='train_set',
        default='squick_coref.conll',  # 'train_orig_all.conll',
        help='Train dataset in conllu format',
        )
parser.add_argument(
        '--test',
        dest='test_set',
        default='squick_coref.conll',
        help='Test dataset in conllu format',
        )
parser.add_argument(
        '--fasttext',
        default=None,
        dest='fasttext',
        help='Fasttext  model to use.'
        )


def save_model(model):
    d = model.state_dict()
    d['hyperparams'] = args
    name = './runs_entity/{}/model_{:09d}.pt'.format(
            args.exp_name, args.word_count
            )
    torch.save(d, name)


def wordvector_for_mention(mention):
    sentence = mention.sentence
    return wordvector[sentence[mention.head].FORM.lower()]


class Entity():
    def __init__(self):
        self.mentions = []
        self.gold_entities = {}
        self.proper_nouns = set()
        self.nouns = set()
        self.pronouns = set()
        self.features = torch.zeros(len(feats_codec.classes_))

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

        self._calculate_features()

    def as_set(self):
        return set(
                [mention.get_identifier() for mention in self.mentions]
                )


def quantized_distance_top_to_mention(entity, mention):
    """
    The distance between the most recent mention of the entity,
    and the query mention.
    dim = 1
    """
    # Distance form the entity
    # assume the mentions are from the same document
    top_mention = entity.mentions[-1]

    distance = abs(mention.sentence.sent_rank - top_mention.sentence.sent_rank)
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
    # TODO limit to subset
    query = to_one_hot(deprel_codec, mention.sentence[mention.head].DEPREL)
    key = to_one_hot(deprel_codec, mention.sentence[mention.head].DEPREL)

    return torch.sum(query * key, dim=0, keepdim=True)


def wordvector_similarity_entity_to_mention(entity, mention):
    """
    Average of the dot product between the wordvector of the mention,
    and each mention in the entity.
    dim = 1
    """
    dots = []
    query = wordvector_for_mention(mention)

    if mention.type() == 'NOMINAL':
        if len(entity.nouns) == 0:
            return torch.zeros(1)
        for key in entity.nouns:
            dots.append(np.dot(query, wordvector[key]))
        return torch.mean(torch.tensor(dots), dim=0, keepdim=True)
    elif mention.type() == 'PROPER':
        if len(entity.proper_nouns) == 0:
            return torch.zeros(1)
        for key in entity.proper_nouns:
            dots.append(np.dot(query, wordvector[key]))
        return torch.mean(torch.tensor(dots), dim=0, keepdim=True)

    # LIST PRONOMIAL
    return torch.zeros(1)


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
    # Role Appositive    [[Actress] Jane Doe]
    #                    same issue as for acronyms
    # Demonym            need lists for Dutch
    # Relative pronoun   [the finance street [which] has already formed
    #                    in the Waitan district])

    top = entity.mentions[-1]
    if mention.sentence != top.sentence:
        return torch.zeros(3)

    ap = 0.0  # appositive
    pn = 0.0  # predicate nominative
    sc = 0.0  # part of same conjunction

    sentence = mention.sentence
    mdr = sentence[mention.head].DEPREL
    tdr = sentence[top.head].DEPREL

    if top.head == mention.head:
        if tdr == 'cop' or mdr == 'cop':
            # predicate nominative
            pn = 1.0
        else:
            pn = 0.0
        if mdr == 'conj' and tdr == 'conj':
            # both members of the same conjunction [A], [B], ....
            sc = 1.0

    if (top.head == sentence[mention.head].ID and tdr == 'conj') or \
       (mention.head == sentence[top.head].ID and mdr == 'conj'):
        # whole list and a member of the list
        sc = 1.0

    # not both part of a list (conjunction)
    if not (tdr == 'conj' and mdr == 'conj'):
        if mention.ids[-1] < top.ids[0]:
            # the order is: mention - top
            separation = top.ids[0] - mention.ids[-1]
            separator = mention.ids[-1] + 1
        else:
            # the order is: top - mention
            separation = mention.ids[0] - top.ids[-1]
            separator = top.ids[-1] + 1

        if separation == 0 or (
                separation == 1 and sentence[separator].DEPREL == 'punct'
                ):
            # directly adjacent or separated by a punctuation
            ap = 1.0

    return torch.tensor([ap, pn, sc])


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

    # Mention type : 4
    mtype = to_one_hot(mention_type_codec, mention.type())

    query = torch.cat([
        torch.tensor([mentions_to_entities_ratio, norm_location]),
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

            #  * 1 wordvector_similarity_entity_to_mention
            wordvector_similarity_entity_to_mention(entity, mention),

            #  * 1 string_match_entity_to_mention
            string_match_entity_to_mention(entity, mention),

            #  * 32 feature match entity
            entity.features * features_for_mention(mention),

            #  * 4 mention type
            to_one_hot(mention_type_codec, mention.type()),

            #  * 1 semantic_role_similiarty_top_to_mention
            semantic_role_similiarty_top_to_mention(entity, mention),

            #  * 1 quantized_distance_top_to_mention
            quantized_distance_top_to_mention(entity, mention)
            ]))

    while len(input) < MAX_CANDIDATES:
        input.append(torch.zeros(41))

    # pass through network MAX_CANDIDATES * 41 -> MAX_CANDIDATES
    input = torch.cat(input)
    return net.combine_evidence(input)


def oracle_losses(entities=[], mention=None):
    """
    Given the entities, and the mention, calculate losses for
    the possible actions.
    """
    # loss for starting a new Entity, filled in below
    losses = [0.]

    total_gold = 0
    max_purity = 0.0
    for entity in entities:
        # loss for adding the mention to this entity
        gold_entity = mention.refid

        if gold_entity in entity.gold_entities:
            same_gold = entity.gold_entities[mention.refid]
        else:
            same_gold = 0
        total_gold += same_gold
        purity = same_gold / len(entity.mentions)

        losses.append(1.0 - purity)
        max_purity = max(max_purity, purity)

    losses[0] = total_gold * max_purity

    return torch.tensor(losses)


def train_one(net, entities=[], mention=None, dynamic_oracle=False):
    net.train()

    # sort entities by distance
    ranking = []
    for entity in entities:
        top = entity.mentions[-1]
        rank = top.sentence.sent_rank
        ranking.append([entity, rank])
    ranking.sort(key=lambda k: -k[1])

    # take top N=10 entities
    if len(ranking) > MAX_CANDIDATES:
        ranking = ranking[0:MAX_CANDIDATES]
    candidates = [rank[0] for rank in ranking]

    # score the most likely action as predicted by our network
    # MAX_CANDIDATES+1 -> MAX_CANDIDATES+1
    picked = torch.cat([
        action_new_probability(net, entities, mention),
        action_add_probabilities(net, candidates, mention)
        ])
    all_probs = net.pick_action(picked)

    # drop entries for non-existant entities
    all_probs = torch.softmax(
            all_probs[0:len(candidates) + 1],
            dim=0
            )

    # pick the most likely action for a dynamic oracle
    dynamic_action = all_probs.argmax().item()

    # ask the oracle for losses for a static oracle
    losses = oracle_losses(candidates, mention)
    static_action = losses.argmin().item()

    # hingeloss
    loss = torch.mean(torch.relu(
            0.05 + all_probs - all_probs[static_action]
            )**2.0)

    if dynamic_oracle:
        action = dynamic_action
    else:
        action = static_action

    if action == 0:
        # start a new entity
        new_entity = Entity()
        new_entity.add(mention)
        entities.append(new_entity)
    else:
        if action > len(candidates):
            logging.error(
                    'Picking an non-existing candidate {}'.format(action)
                    )
        else:
            # add to existing entity
            candidates[action - 1].add(mention)

    return loss


def eval(net, doc):
    net.eval()
    trace = ''

    # start without entities
    entities = []

    # add the mentions one-by-one to the entities
    for mention in doc:

        # sort entities by distance
        ranking = []
        for entity in entities:
            top = entity.mentions[-1]
            rank = top.sentence.sent_rank
            ranking.append([entity, rank])
        ranking.sort(key=lambda k: -k[1])

        # take top N=10 entities
        if len(ranking) > MAX_CANDIDATES:
            ranking = ranking[0:MAX_CANDIDATES]
        candidates = [rank[0] for rank in ranking]

        # score the most likely action as predicted by our network
        # MAX_CANDIDATES+1 -> MAX_CANDIDATES+1
        picked = torch.cat([
            action_new_probability(net, entities, mention),
            action_add_probabilities(net, candidates, mention)
            ])
        all_probs = net.pick_action(picked)

        all_probs = all_probs[0:len(candidates) + 1]
        action = all_probs.argmax().item()

        if action == 0:
            # start a new entity
            new_entity = Entity()
            new_entity.add(mention)
            new_entity.rank = len(entities)
            entities.append(new_entity)
            trace += ' {}N'.format(new_entity.rank)
        else:
            # add to existing entity
            existing_entity = candidates[action - 1]
            existing_entity.add(mention)
            trace += ' {} '.format(existing_entity.rank)
    logging.info(trace)

    # score the entities
    # build list of sets for both gold and system
    gold_sets = {}
    for mention in doc:
        if mention.refid not in gold_sets:
            gold_sets[mention.refid] = set()
        gold_sets[mention.refid].add(mention.get_identifier())
    gold_sets = list(gold_sets.values())

    system_sets = []
    for entity in entities:
        system_sets.append(entity.as_set())

    score_muc = muc(gold_sets, system_sets)
    score_b3 = b_cubed(gold_sets, system_sets)
    score_ce = ceaf_e(gold_sets, system_sets)
    logging.info('\nMuc: {}\n B3:  {}\n Ce:  {}\nCnl: {}'.format(
        score_muc, score_b3, score_ce, [
            (score_muc[0] + score_b3[0] + score_ce[0]) / 3.,
            (score_muc[1] + score_b3[1] + score_ce[1]) / 3.,
            (score_muc[2] + score_b3[2] + score_ce[2]) / 3.
            ]))
    return score_muc, score_b3, score_ce


def train(net, test_docs, train_docs,
          learning_rate=1e-3, epochs=1):

    sampler = RandomSampler(train_docs)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        )

    docs_per_eval = 25
    docs_to_go = docs_per_eval

    for epoch in range(epochs):
        for doc_rank in sampler:
            doc = train_docs[doc_rank]
            logging.info('Training doc has length {}'.format(len(doc)))

            if len(doc) == 0:
                continue

            # start without entities
            entities = []

            # add the mentions one-by-one to the entities
            total_loss = 0
            for mention in doc:
                total_loss += train_one(net, entities, mention)

            args.word_count += 1
            total_loss = total_loss / len(doc)
            writer.add_scalar('total_loss', total_loss.item(), args.word_count)

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()

                # update parameters
                optimizer.step()

            for name, param in net.state_dict().items():
                writer.add_scalar(
                        'norm_' + name,
                        torch.norm(param.float()),
                        args.word_count
                        )

            if docs_to_go == 1:
                # eval
                print('Evaluatig epoch', epoch)
                score_muc = np.zeros(3)
                score_b3 = np.zeros(3)
                score_ce = np.zeros(3)
                for i in range(len(test_docs)):
                    one_muc, one_b3, one_ce = eval(net, test_docs[i])
                    score_muc[0] += one_muc[0]
                    score_muc[1] += one_muc[1]
                    score_muc[2] += one_muc[2]
                    score_b3[0] += one_b3[0]
                    score_b3[1] += one_b3[1]
                    score_b3[2] += one_b3[2]
                    score_ce[0] += one_ce[0]
                    score_ce[1] += one_ce[1]
                    score_ce[2] += one_ce[2]

                score_muc /= len(test_docs)
                score_b3 /= len(test_docs)
                score_ce /= len(test_docs)

                writer.add_scalar('muc_p', score_muc[0], args.word_count)
                writer.add_scalar('muc_r', score_muc[1], args.word_count)
                writer.add_scalar('muc_f', score_muc[2], args.word_count)

                writer.add_scalar('b3_p', score_b3[0], args.word_count)
                writer.add_scalar('b3_r', score_b3[1], args.word_count)
                writer.add_scalar('b3_f', score_b3[2], args.word_count)

                writer.add_scalar('ce_p', score_ce[0], args.word_count)
                writer.add_scalar('ce_r', score_ce[1], args.word_count)
                writer.add_scalar('ce_f', score_ce[2], args.word_count)

                writer.add_scalar('conll',
                                  (score_muc[2] + score_b3[2] + score_ce[2])/3,
                                  args.word_count)
                docs_to_go = docs_per_eval
                save_model(net)
            else:
                docs_to_go -= 1


def main(args):
    global writer
    global wordvector

    logging.basicConfig(level=logging.INFO)

    exp_name = 'entity' + \
        '_v0.10'

    args.exp_name = exp_name
    print('Experiment {}'.format(args.exp_name))

    # Train dataset
    train_set = ConlluDataset(args.train_set)

    train_mentions = []
    for sentence in train_set:
        _, mentions = preprocess_sentence(sentence)
        train_mentions += mentions

    train_docs = [[] for i in train_set.doc_lengths]
    for mention in train_mentions:
        sentence = mention.sentence
        doc_rank = sentence.doc_rank
        train_docs[doc_rank].append(mention)

    logging.info('Train dataset contains {} documents, {} sentences'.format(
        len(train_docs), len(train_set)
        ))

    # Test dataset
    test_set = ConlluDataset(args.test_set)

    test_mentions = []
    for sentence in test_set:
        _, mentions = preprocess_sentence(sentence)
        test_mentions += mentions

    test_docs = [[] for i in test_set.doc_lengths]
    for mention in test_mentions:
        sentence = mention.sentence
        doc_rank = sentence.doc_rank
        test_docs[doc_rank].append(mention)

    logging.info('Test dataset contains {} documents, {} sentences'.format(
        len(test_docs), len(test_set)
        ))

    # Load the FastText model
    wordvector = fasttext.load_model(
            '/home/jiska/Code/ernie/resources/fasttext.model.bin')

    print('Tensorboard output in "{}".'.format(exp_name))
    writer = SummaryWriter('runs_entity/' + exp_name)

    print('Ctrl-c will abort training and save the current model.')

    def sigterm_handler(_signo, _stack_frame):
        global writer

        writer.close()
        # save_model(net)
        print('Ctrl-c detected, aborting')
        exit(0)

    # signal.signal(signal.SIGTERM, sigterm_handler)
    # signal.signal(signal.SIGINT, sigterm_handler)

    net = EntityNet()

    args.word_count = 0
    train(net, test_docs, train_docs, epochs=1000)

    save_model(net)
    writer.close()


if __name__ == '__main__':
    torch.manual_seed(43)
    args = parser.parse_args()
    main(args)

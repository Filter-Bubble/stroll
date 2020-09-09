import argparse
import logging

import fasttext
import numpy as np
from math import log

import torch

from numba import jit
from scorch.scores import muc, b_cubed, ceaf_e

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence
from stroll.model import EntityNet

from stroll.entity import MAX_CANDIDATES
from stroll.entity import Entity
from stroll.entity import action_new_probability, action_add_probabilities
from stroll.entity import set_wordvector

# Global arguments for dealing with Ctrl-C
global writer
writer = None

parser = argparse.ArgumentParser(
        description='Train an entity centric transition based coreference net'
        )
parser.add_argument(
        '--train',
        dest='train_set',
        default='squick_coref2.conll',  # 'train_orig_all.conll',
        help='Train dataset in conllu format',
        )
parser.add_argument(
        '--test',
        dest='test_set',
        default='squick_coref2.conll',
        help='Test dataset in conllu format',
        )
parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        type=float,
        default=1e-3,
        help='Test dataset in conllu format',
        )
parser.add_argument(
        '--fasttext',
        default=None,
        dest='fasttext',
        help='Fasttext  model to use.'
        )


def save_model(model, optimizer):
    d = model.state_dict()
    d['hyperparams'] = args
    d['optimizer'] = optimizer.state_dict()
    name = './runs_entity2/{}/model_{:09d}.pt'.format(
            args.exp_name, args.word_count
            )
    torch.save(d, name)


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


def train_one(net, entities=[], mention=None, dynamic_oracle=True):
    net.train()

    # short cut for single action
    if len(entities) == 0:
        new_entity = Entity()
        new_entity.refid = len(entities)
        new_entity.add(mention)
        entities.append(new_entity)
        return {
                'truth': mention.refid,
                'guess': 'new',
                'options': ['new'],
                'probs': torch.cat([torch.ones(1)])
                }

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
    all_probs = torch.cat([
        action_new_probability(net, entities, mention),
        action_add_probabilities(net, candidates, mention)
        ])

    # pick the most likely action for a dynamic oracle
    dynamic_action = all_probs.argmax().item()

    # ask the oracle for costs
    costs = oracle_losses(candidates, mention)
    oracle_action = costs.argmin().item()

    if dynamic_oracle:
        action = dynamic_action
    else:
        action = oracle_action

    # truth mention.refid
    # guess action
    # actions: ['new', ...candidates.refid]
    # probs: all_probs
    trace = {
            'truth': mention.refid,

            'guess': 'new' if action == 0 else candidates[action-1].refid,

            'options':
            ['new'] + [candidate.refid for candidate in candidates],

            'probs': all_probs
            }

    # print('Oracle:', oracle_action, 'Action:', dynamic_action, 'Loss:', loss)
    if action == 0:
        # start a new entity
        new_entity = Entity()
        new_entity.refid = len(entities)
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

    return trace


def eval(net, doc):
    net.eval()
    trace = ''

    # start without entities
    entities = []

    # add the mentions one-by-one to the entities
    for mention in doc:
        # short cut for single action
        if len(entities) == 0:
            new_entity = Entity()
            new_entity.refid = len(entities)
            new_entity.rank = len(entities)
            new_entity.add(mention)
            entities.append(new_entity)
            continue

        # sort entities by distance
        ranking = []
        for entity in entities:
            top = entity.mentions[-1]
            rank = top.sentence.sent_rank
            ranking.append([entity, rank])
        ranking.sort(key=lambda k: -k[1])

        # take top N entities
        if len(ranking) > MAX_CANDIDATES:
            ranking = ranking[0:MAX_CANDIDATES]
        candidates = [rank[0] for rank in ranking]

        # score the most likely action as predicted by our network
        all_probs = torch.cat([
            action_new_probability(net, entities, mention),
            action_add_probabilities(net, candidates, mention)
            ])
        action = all_probs.argmax().item()

        if action == 0:
            # start a new entity
            new_entity = Entity()
            new_entity.refid = len(entities)
            new_entity.add(mention)
            new_entity.rank = len(entities)
            entities.append(new_entity)
            trace += ' {} '.format(new_entity.rank)
        else:
            # add to existing entity
            existing_entity = candidates[action - 1]
            existing_entity.add(mention)
            trace += ' {}L'.format(existing_entity.rank)
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


@jit(nopython=True)
def vi_from_cm(cm):
    ni, nj = cm.shape

    n = 0.0
    pa = np.zeros(ni)
    pb = np.zeros(nj)
    for i in range(ni):
        for j in range(nj):
            pa[i] += cm[i, j]
            pb[j] += cm[i, j]
            n += cm[i, j]

    for i in range(ni):
        if pa[i] > 1:
            pa[i] = log(pa[i])
        else:
            pa[i] = 0

    for j in range(nj):
        if pb[j] > 1:
            pb[j] = log(pb[j])
        else:
            pb[j] = 0

    logc = np.zeros_like(cm)
    for i in range(ni):
        for j in range(nj):
            if cm[i, j] > 1:
                logc[i, j] = log(cm[i, j])

    s = 0.0
    for i in range(ni):
        for j in range(nj):
            s += cm[i, j] * (2.0 * logc[i, j] - pa[i] - pb[j])

    return -(s / (n * log(n)))


def vi_from_cm_slow(cm):
    n = cm.sum()

    pa = cm.sum(axis=1)
    pb = cm.sum(axis=0)

    pa = np.where(pa < 1, 1, pa)
    pb = np.where(pb < 1, 1, pb)

    logpa = np.broadcast_to(np.log(pa), (len(pb), len(pa))).transpose()
    logpb = np.broadcast_to(np.log(pb), (len(pa), len(pb)))

    logc = np.log(np.where(cm < 1, 1, cm))

    return -np.sum(cm * (2 * logc - logpa - logpb)) / (n * np.log(n))


def train(net, test_docs, train_docs,
          optimizer, epochs=1):

    sampler = RandomSampler(train_docs)
    docs_per_eval = 25
    docs_to_go = docs_per_eval

    for epoch in range(epochs):
        for doc_rank in sampler:
            doc = train_docs[doc_rank]
            logging.info('Training doc has length {}'.format(len(doc)))

            if len(doc) == 0 or len(doc) > 500:
                continue

            # start without entities
            entities = []

            # add the mentions one-by-one to the entities
            trace = []
            ids_truth = set()
            ids_guess = set()
            next_refid = 0
            for mention in doc:
                step = train_one(net, entities, mention)
                trace.append(step)

                # find all correct / guessed refids
                guess = step['guess']
                if guess == 'new':
                    guess = next_refid
                    next_refid += 1
                ids_guess.add(guess)
                ids_truth.add(step['truth'])

            # map refid's to an index
            truth_to_idx = {}
            for idx, t in enumerate(ids_truth):
                truth_to_idx[t] = idx

            guess_to_idx = {}
            ids_guess.add('singleton')  # a guaranteed-to-be singleton entity
            for idx, t in enumerate(ids_guess):
                guess_to_idx[t] = idx

            # build contigency matrix
            cm = np.zeros([len(ids_truth), len(ids_guess)])
            next_refid = 0
            for step in trace:
                guess = step['guess']
                if guess == 'new':
                    guess = next_refid
                    next_refid += 1
                truth = step['truth']
                cm[truth_to_idx[truth], guess_to_idx[guess]] += 1

            total_loss = torch.zeros(1)

            # calculate losses
            next_refid = 0
            for step in trace:
                guess = step['guess']
                if guess == 'new':
                    guess = next_refid
                    next_refid += 1
                truth = step['truth']

                # remove this step
                cm[truth_to_idx[truth], guess_to_idx[guess]] -= 1

                # calculate the VI for possible alternative steps
                VI_by_alt = {}
                for alt in ids_guess:
                    # if alt in step['options'] or \
                    #         alt == 'singleton' or \
                    #         cm[truth_to_idx[truth], guess_to_idx[alt]] != 0:
                    # take this step
                    cm[truth_to_idx[truth], guess_to_idx[alt]] += 1

                    d = vi_from_cm(cm)
                    VI_by_alt[alt] = np.exp(10. * d) * d

                    # undo this step
                    cm[truth_to_idx[truth], guess_to_idx[alt]] -= 1

                # softmax over probs
                all_probs = torch.softmax(step['probs'], dim=0)

                vi = torch.zeros_like(all_probs)
                for i, alt in enumerate(step['options']):
                    if alt == 'new':
                        continue
                    # for this option, the cost is the VI as calculated
                    vi[i] = VI_by_alt.pop(step['options'][i])

                # all remaining choices / clusters are reached via starting
                # a new cluster, so take the minimum value
                vi[0] = min(VI_by_alt.values())

                total_loss += torch.dot(vi, all_probs)

                # restore the step
                cm[truth_to_idx[truth], guess_to_idx[guess]] += 1

            args.word_count += 1

            if total_loss.requires_grad and \
                    not torch.any(torch.isnan(total_loss)):
                total_loss = total_loss / len(doc)
                optimizer.zero_grad()
                total_loss.backward()

                # update parameters
                optimizer.step()
            else:
                logging.warning('Loss is NaN or does not requires gradient')
                optimizer.zero_grad()

            writer.add_scalar(
                    'total_loss',
                    total_loss.item(),
                    args.word_count
                    )
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

                writer.add_scalar('muc_r', score_muc[0], args.word_count)
                writer.add_scalar('muc_p', score_muc[1], args.word_count)
                writer.add_scalar('muc_f', score_muc[2], args.word_count)

                writer.add_scalar('b3_r', score_b3[0], args.word_count)
                writer.add_scalar('b3_p', score_b3[1], args.word_count)
                writer.add_scalar('b3_f', score_b3[2], args.word_count)

                writer.add_scalar('ce_r', score_ce[0], args.word_count)
                writer.add_scalar('ce_p', score_ce[1], args.word_count)
                writer.add_scalar('ce_f', score_ce[2], args.word_count)

                writer.add_scalar('conll',
                                  (score_muc[2] + score_b3[2] + score_ce[2])/3,
                                  args.word_count)
                docs_to_go = docs_per_eval
                save_model(net, optimizer)
            else:
                docs_to_go -= 1


def main(args):
    global writer
    global wordvector

    logging.basicConfig(level=logging.INFO)

    exp_name = 'entity' + \
        '_v2.6_dyn_{}_stat'.format(MAX_CANDIDATES)

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
    set_wordvector(fasttext.load_model(
            '/home/jiska/Code/ernie/resources/fasttext.model.bin'
            ))

    print('Tensorboard output in "{}".'.format(exp_name))
    writer = SummaryWriter('runs_entity2/' + exp_name)

    net = EntityNet()

    print('Looking for "restart.pt".')
    try:
        restart = torch.load('restart.pt')
        net.load_state_dict(restart, strict=False)
        logging.info('Restart succesful.')
        args.word_count = restart['hyperparams'].word_count
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.learning_rate,
            )
        optimizer.load_state_dict(restart['optimizer'])

    except(FileNotFoundError):
        logging.info('Restart failed, starting from scratch.')
        args.word_count = 0
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.learning_rate,
            )

    args.word_count = 0
    train(net, test_docs, train_docs, optimizer, epochs=1000)
    writer.close()


if __name__ == '__main__':
    torch.manual_seed(42)
    args = parser.parse_args()
    main(args)  # BUGFIX after run 32

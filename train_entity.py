import sys
import signal
import argparse
import logging

import fasttext
import numpy as np
import torch

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

global log_oracle
log_oracle = np.zeros(MAX_CANDIDATES + 1)
global log_action
log_action = np.zeros(MAX_CANDIDATES + 1)

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
    name = './runs_entity/{}/model_{:09d}.pt'.format(
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


def train_one(net, entities=[], mention=None, dynamic_oracle=False):
    global log_action
    global log_oracle
    net.train()

    # short cut for single action
    # this prevents blowing up the new_entity_prob in the backward pass
    if len(entities) == 0:
        new_entity = Entity()
        new_entity.add(mention)
        entities.append(new_entity)
        loss = torch.zeros(1, requires_grad=True)
        return loss

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
    all_probs = all_probs[0:len(candidates) + 1]

    # pick the most likely action for a dynamic oracle
    dynamic_action = all_probs.argmax().item()
    log_action[dynamic_action] += 1.0

    # ask the oracle for costs
    costs = oracle_losses(candidates, mention)
    oracle_action = costs.argmin().item()
    log_oracle[oracle_action] += 1.0

    # hingeloss
    # all_probs = torch.softmax(all_probs, dim=0)
    # loss = torch.mean(torch.relu(
    #         0.05 + all_probs - all_probs[oracle_action]
    #         )**2.0)
    log_all_probs = torch.log_softmax(all_probs, dim=0)
    prob = torch.exp(log_all_probs[oracle_action])
    loss = -1.0 * ((1.0 - prob)**1.5) * log_all_probs[oracle_action]

    # # boost loss for non-new entity
    # if oracle_action == 0:
    #     loss = loss * 0.01

    if dynamic_oracle:
        action = dynamic_action
    else:
        action = oracle_action

    # print('Oracle:', oracle_action, 'Action:', dynamic_action, 'Loss:', loss)
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


def train(net, test_docs, train_docs,
          optimizer, epochs=1):

    sampler = RandomSampler(train_docs)
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
                total_loss += train_one(
                        net, entities, mention
                        )

            args.word_count += 1

            if total_loss.requires_grad:
                total_loss = total_loss / len(doc)
                optimizer.zero_grad()
                total_loss.backward()

                # update parameters
                optimizer.step()

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

                writer.add_scalar('muc_p', score_muc[0], args.word_count)
                writer.add_scalar('muc_r', score_muc[1], args.word_count)
                writer.add_scalar('muc_f', score_muc[2], args.word_count)

                writer.add_scalar('b3_p', score_b3[0], args.word_count)
                writer.add_scalar('b3_r', score_b3[1], args.word_count)
                writer.add_scalar('b3_f', score_b3[2], args.word_count)

                writer.add_scalar('ce_p', score_ce[0], args.word_count)
                writer.add_scalar('ce_r', score_ce[1], args.word_count)
                writer.add_scalar('ce_f', score_ce[2], args.word_count)
                print('Network:', log_action / np.sum(log_action))
                print('Oracle:', log_oracle / np.sum(log_oracle))

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
        '_v0.34_stat'.format(MAX_CANDIDATES)

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
    writer = SummaryWriter('runs_entity/' + exp_name)

    print('Ctrl-c will abort training and save the current model.')

    def sigterm_handler(_signo, _stack_frame):
        global writer

        writer.close()
        print('Ctrl-c detected, aborting')
        exit(0)

    # signal.signal(signal.SIGTERM, sigterm_handler)
    # signal.signal(signal.SIGINT, sigterm_handler)

    net = EntityNet(max_candidates=MAX_CANDIDATES)

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
    torch.manual_seed(43)
    args = parser.parse_args()
    main(args)  # BUGFIX after run 32

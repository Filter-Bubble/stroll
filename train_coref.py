import sys
import time
import signal
import argparse
import logging

import numpy as np
import math

import torch

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster

import dgl

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence
from stroll.coref import get_mentions
from stroll.coref import features_mention, features_mention_pair
from stroll.graph import GraphDataset
from stroll.model import CorefNet
from stroll.labels import FasttextEncoder

MAX_MENTION_DISTANCE = 50
MAX_MENTION_PER_DOC = 1000


# Global arguments for dealing with Ctrl-C
global writer
writer = None

torch.manual_seed(43)


class RandomBatchSampler:
    """
    Randomly sample batches; but keep elements within a batch consecutive.
    """
    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size

        # find the number of batches
        self.nbatches = math.ceil(length / batch_size)

        # create a random sampler over these batches
        self.random_sampler = RandomSampler(range(self.nbatches))

    def __len__(self):
        return self.nbatches

    def __iter__(self):
        self.it = self.random_sampler.__iter__()
        return self

    def __next__(self):
        batch = next(self.it)

        start = batch * self.batch_size
        end = min(start + self.batch_size, self.length)
        return range(start, end)


def get_optimizer_and_scheduler_for_net(
        net,
        solver='CE',
        learning_rate=1e-2,
        ):
    if solver == 'SGD':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            momentum=0.9
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.9
            )
    elif solver == 'ADAM':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate,
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lambda epoch: 0.9**min(max(0, (epoch - 4) // 2), 36)],
            )
    elif solver == 'ADAMW':
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=learning_rate
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=1.0
            )
    else:
        print('Solver not implemented.')
        sys.exit(-1)

    return optimizer, scheduler


def predict_clusters(similarities, nlinks=200, word_count=0):
    # take the upper diagonal as needed for linkage
    fulldist = similarities.numpy()
    fulldist = fulldist[np.triu_indices(len(fulldist), 1)]

    # turn similarities into distances
    fulldist = np.nan_to_num(np.exp(-1. * fulldist))

    Z = linkage(fulldist, 'single')

    return list(fcluster(Z, Z[nlinks, 2], criterion='distance'))


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
      similarities   torch.tensor(nmentions, nmetsions)
      link           torch.tensor(nmentions, nmentions)
    """

    nmentions = len(mentions)
    links = torch.zeros([nmentions, nmentions])
    # BUG: oeps, very close to 0
    similarities = torch.ones([nmentions, nmentions]) * -1e-8

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
                gvec[aid].view(-1),
                features_mention(antecedent),
                features_mention_pair(
                    antecedent,
                    mention)
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


def contrastive_loss(links, similarities, tau=torch.tensor(0.7)):
    loss = torch.tensor(0.)

    # build similarity matrix S[i, j] = exp(sim(i,j)/\tau)(1-\delta(i, j))
    S = torch.exp(similarities / tau)

    # build link matrix L[i, j] == 1 iff mentions are linked, 0 else
    L = links

    nmentions = len(links)
    for i in range(nmentions):
        same_label = torch.nonzero(L[i, :])
        if len(same_label) < 2:
            continue
        contrast = torch.sum(S[i, :])

        loss += -1.0 / (len(same_label) - 1) * \
            torch.sum(
                    torch.log(
                        S[i, same_label] / contrast
                    )
            )

    return loss / nmentions


def train(net, trainloader,
          test_graph, test_mentions, test_clusters,
          optimizer, scheduler,
          epochs=60):
    global writer

    # diagnostic settings
    count_per_eval = 25000
    next_eval = count_per_eval
    t0 = time.time()
    best_model_score = 0.
    word_count = args.word_count

    print('Start training for {:d} epochs.'.format(epochs))

    # loop over the epochs
    for epoch in range(epochs):
        writer.add_scalar(
                'learning_rate',
                optimizer.param_groups[0]['lr'],
                word_count
                )
        # loop over each minibatch
        for train_graph, mentions in trainloader:
            net.train()

            # predict mentions, and vectors
            id_out, gvec = net(train_graph)

            # predict coreference pairs:

            # correct mentions
            target = train_graph.ndata['coref'].view(-1).clamp(0, 1)
            target.detach()

            # take the indices of the nodes that are gold-mentions
            mention_idxs = torch.nonzero(target)

            if len(mention_idxs) > 0:
                links, similarities = predict_similarities(
                        net,
                        mentions,
                        gvec[mention_idxs]
                        )

                loss_sim = contrastive_loss(links, similarities)
            else:
                loss_sim = torch.tensor(0)

            # apply loss
            loss_total = loss_sim   # + loss_id
            optimizer.zero_grad()

            # for batches without a mention,  the loss is zero and
            # loss.backward() raises a RuntimeError
            try:
                loss_total.backward()
                optimizer.step()
            except RuntimeError as e:
                logging.error(
                        'Loss={} ignoring runtime error in torch: {}'.format(
                            loss_total.item(), e))

            # diagnostics
            word_count += len(train_graph)
            args.word_count = word_count
            writer.add_scalar('loss_sim', loss_sim.item(), word_count)
            writer.add_scalar('loss_total', loss_total.item(), word_count)

            if word_count > next_eval:
                dur = time.time() - t0
                net.eval()
                with torch.no_grad():
                    id_out, gvec = net(test_graph)

                    # coreference pairs: score clustering on gold mentions
                    target = test_graph.ndata['coref'].view(-1).clamp(0, 1)
                    mention_idxs = torch.nonzero(target)

                    links, similarities = predict_similarities(
                            net,
                            test_mentions,
                            gvec[mention_idxs]
                            )

                    system_clusters = predict_clusters(
                            similarities,
                            nlinks=750,
                            word_count=word_count
                            )

                    score_sim_ar = adjusted_rand_score(
                            test_clusters, system_clusters
                            )
                    score_sim_ami = adjusted_mutual_info_score(
                            test_clusters, system_clusters
                            )

                # Report
                print('Elements {:08d} |'.format(word_count),
                      'AR {:.4f}|'.format(score_sim_ar),
                      'words/sec {:4.3f}'.format(count_per_eval / dur)
                      )

                writer.add_scalar('s_sim_ar', score_sim_ar, word_count)
                writer.add_scalar('s_sim_ami', score_sim_ami, word_count)

                for name, param in net.state_dict().items():
                    writer.add_scalar(
                            'norm_' + name,
                            torch.norm(param.float()),
                            word_count
                            )

                # Save best-until-now model
                score = score_sim_ar
                if epoch > 0 and score > best_model_score:
                    logging.info('Saving new best model at step {:09d}'.format(
                        word_count
                        ))
                    best_model_score = score
                    save_model(net)

                # reset timer
                next_eval = next_eval + count_per_eval
                t0 = time.time()

        save_model(net)
        print('Epoch {} done'.format(epoch))

        writer.add_scalar(
                'learning_rate',
                optimizer.param_groups[0]['lr'],
                word_count-1
                )
        scheduler.step()


def save_model(model):
    d = model.state_dict()
    d['hyperparams'] = args
    name = './runs/{}/model_{:09d}.pt'.format(args.exp_name, args.word_count)
    torch.save(d, name)


parser = argparse.ArgumentParser(
        description='Train a R-GCN for Coreference resolution.'
        )
parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=60,
        help='Number of epochs to train'
        )
parser.add_argument(
        '--h_dims',
        dest='h_dims',
        type=int,
        default=64,
        help='Dimension of the hidden states'
        )
parser.add_argument(
        '--batch_size',
        dest='batch_size',
        type=int,
        default=150,
        help='Evaluation batch size.'
        )
parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        type=float,
        default='1e-3',
        help='Initial learning rate.'
        )
parser.add_argument(
        '--features',
        nargs='*',
        dest='features',
        default=['UPOS', 'FEATS', 'DEPREL', 'WVEC'],
        choices=['UPOS', 'XPOS', 'FEATS', 'DEPREL', 'WVEC'],
        help='Features used by the model'
        )
parser.add_argument(
        '--fasttext',
        default=None,
        dest='fasttext',
        help='Fasttext  model to use instead of Bert.'
        )
parser.add_argument(
        '--h_layers',
        dest='h_layers',
        type=int,
        default=2,
        help='Number of hidden RGCN layers.',
        )
parser.add_argument(
        '--solver',
        dest='solver',
        default='ADAM',
        choices=['ADAM', 'SGD', 'ADAMW'],
        help='Optimizer (SGD/ADAM) and learning rate schedule',
        )
parser.add_argument(
        '--train',
        dest='train_set',
        default='train.conllu',
        help='Train dataset in conllu format',
        )
parser.add_argument(
        '--test',
        dest='test_set',
        default='quick.conllu',
        help='Test dataset in conllu format',
        )


def coref_collate(batch):
    """
    Collate function to batch samples together.
    """

    mentions = []
    for g in batch:
        mentions += get_mentions(g.sentence)
    return dgl.batch(batch), mentions


def main(args):
    global writer

    logging.basicConfig(level=logging.INFO)

    exp_name = args.solver + \
        'v3.32' + \
        '_{:1.0e}'.format(args.learning_rate) + \
        '_{:d}b'.format(args.batch_size) + \
        '_HL' + \
        '_{:d}d'.format(args.h_dims) + \
        '_{:d}l'.format(args.h_layers) + \
        '_'.join(args.features)

    if 'WVEC' in args.features:
        sentence_encoder = FasttextEncoder(args.fasttext)
        exp_name += '_' + sentence_encoder.name
    else:
        sentence_encoder = None

    args.exp_name = exp_name
    print('Experiment {}'.format(args.exp_name))

    logging.info(
            'Preparing train dataset {}'.format(args.train_set)
            )

    train_raw = ConlluDataset(args.train_set)
    for sentence in train_raw:
        preprocess_sentence(sentence)

    train_set = GraphDataset(
            dataset=train_raw,
            sentence_encoder=sentence_encoder,
            features=args.features
            )
    trainloader = DataLoader(
        train_set,
        batch_sampler=RandomBatchSampler(len(train_set), args.batch_size),
        # num_workers=2,
        collate_fn=coref_collate
        )

    logging.info(
            'Building test graph from {}.'.format(args.test_set)
            )
    test_raw = ConlluDataset(args.test_set)
    for sentence in test_raw:
        preprocess_sentence(sentence)

    test_set = GraphDataset(
            dataset=test_raw,
            sentence_encoder=sentence_encoder,
            features=args.features
            )
    testloader = DataLoader(
        test_set,
        num_workers=2,
        batch_size=20,
        collate_fn=coref_collate
        )

    test_graph = []
    test_clusters = []

    test_mentions = []
    for g, m in testloader:
        test_graph.append(g)
        test_mentions += m

    test_graph = dgl.batch(test_graph)
    for mention in test_mentions:
        sentence = mention.sentence
        test_clusters.append(
                int(mention.refid) + sentence.doc_rank * MAX_MENTION_PER_DOC
                )

    logging.info('Building model.')
    net = CorefNet(
        in_feats=train_set.in_feats,
        in_feats_b=(args.h_dims + 7) * 2 + 5,
        h_layers=args.h_layers,
        h_dims=args.h_dims,
        activation='relu'
        )
    logging.info(net.__repr__())

    logging.info('Initializing Optimizer and learning rate scheduler.')
    optimizer, scheduler = get_optimizer_and_scheduler_for_net(
            net,
            args.solver,
            args.learning_rate
            )

    print('Looking for "restart.pt".')
    try:
        restart = torch.load('restart.pt')
        net.load_state_dict(restart, strict=False)
        logging.info('Restart succesful.')
        args.word_count = restart['hyperparams'].word_count
    except(FileNotFoundError):
        logging.info('Restart failed, starting from scratch.')
        args.word_count = 0

    print('Tensorboard output in "{}".'.format(exp_name))
    writer = SummaryWriter('runs/' + exp_name)

    print('Ctrl-c will abort training and save the current model.')

    def sigterm_handler(_signo, _stack_frame):
        global writer

        writer.close()
        save_model(net)
        print('Ctrl-c detected, aborting')
        exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    print(net)
    train(net,
          trainloader,
          test_graph,
          test_mentions,
          test_clusters,
          optimizer,
          scheduler,
          args.epochs
          )

    save_model(net)
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

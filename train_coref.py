import time
import signal
import argparse
import logging

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_recall_fscore_support as PRF

import dgl

from scorch.scores import muc

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence
from stroll.coref import mark_gold_anaphores
from stroll.coref import nearest_linking
from stroll.coref import predict_anaphores, predict_similarities
from stroll.coref import coref_collate
from stroll.graph import GraphDataset
from stroll.model import CorefNet
from stroll.loss import contrastive_loss
from stroll.labels import FasttextEncoder
from stroll.evaluate import clusters_to_sets
from stroll.train import get_optimizer_and_scheduler_for_net
from stroll.train import RandomBatchSampler

MAX_MENTION_DISTANCE = 20
MAX_MENTION_PER_DOC = 1000


# Global arguments for dealing with Ctrl-C
global writer
writer = None

torch.manual_seed(43)


def train(net, trainloader,
          test_graph, test_mentions, test_clusters,
          optimizer, scheduler,
          epochs=60):
    global writer

    # diagnostic settings
    count_per_eval = 80000
    next_eval = count_per_eval
    t0 = time.time()
    best_ana_score = 0.
    best_muc_score = 0.

    word_count = args.word_count

    anaphore_loss = torch.nn.BCEWithLogitsLoss()

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
            gvec = net(train_graph)

            # predict coreference pairs:

            # correct mentions
            target = train_graph.ndata['coref'].view(-1).clamp(0, 1)

            # take the indices of the nodes that are gold-mentions
            mention_idxs = torch.nonzero(target)

            if len(mention_idxs) > 0:
                links, similarities = predict_similarities(
                        net,
                        mentions,
                        gvec[mention_idxs]
                        )

                loss_sim = contrastive_loss(links, similarities)

                anaphores = predict_anaphores(
                        net,
                        mentions
                        )

                targets = torch.tensor([
                    mention.anaphore for mention in mentions
                    ])

                loss_ana = anaphore_loss(
                        anaphores.view(-1),
                        targets.view(-1)
                        )
            else:
                loss_ana = torch.tensor(0)
                loss_sim = torch.tensor(0)

            # apply loss
            loss_total = loss_sim + loss_ana
            optimizer.zero_grad()

            # for batches without a mention, the loss is zero and
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
            writer.add_scalar('loss_ana', loss_ana.item(), word_count)
            writer.add_scalar('loss_total', loss_total.item(), word_count)

            if word_count > next_eval:
                dur = time.time() - t0
                net.eval()
                with torch.no_grad():
                    gvec = net(test_graph)

                    target = test_graph.ndata['coref'].view(-1).clamp(0, 1)
                    mention_idxs = torch.nonzero(target)

                    links, similarities = predict_similarities(
                            net,
                            test_mentions,
                            gvec[mention_idxs]
                            )

                    # predict anaphores
                    anaphores = torch.sigmoid(predict_anaphores(
                            net, test_mentions
                            ))

                    # cluster using the predictions
                    system_clusters = nearest_linking(
                            similarities, anaphores
                            )

                    # score the clustering
                    system_set = clusters_to_sets(system_clusters)
                    gold_set = clusters_to_sets(test_clusters)

                    muc_prf = muc(gold_set, system_set)

                    # score the anaphores
                    targets = torch.tensor([
                        mention.anaphore for mention in test_mentions
                        ])

                    ana_scores = PRF(
                            targets,
                            torch.round(anaphores),
                            labels=[1.0]
                            )

                # Report
                print('Elements {:08d} |'.format(word_count),
                      'words/sec {:4.3f}'.format(count_per_eval / dur)
                      )

                writer.add_scalar('s_ana_p', ana_scores[0], word_count)
                writer.add_scalar('s_ana_r', ana_scores[1], word_count)
                writer.add_scalar('s_ana_f', ana_scores[2], word_count)
                writer.add_scalar('s_muc_p', muc_prf[0], word_count)
                writer.add_scalar('s_muc_r', muc_prf[1], word_count)
                writer.add_scalar('s_muc_f', muc_prf[2], word_count)

                for name, param in net.state_dict().items():
                    writer.add_scalar(
                            'norm_' + name,
                            torch.norm(param.float()),
                            word_count
                            )

                # Save best-until-now model
                if epoch > 0 and (
                                 ana_scores[2] > best_ana_score or
                                 muc_prf[2] > best_muc_score
                                 ):
                    if ana_scores[2] > best_ana_score:
                        best_ana_score = ana_scores[2]
                    if muc_prf[2] > best_muc_score:
                        best_muc_score = muc_prf[2]

                    logging.info('Saving new best model at step {:09d}'.format(
                        word_count
                        ))
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
        if scheduler:
            scheduler.step()


def save_model(model):
    d = model.state_dict()
    d['hyperparams'] = args
    name = './runs_coref/{}/model_{:09d}.pt'.format(
            args.exp_name, args.word_count
            )
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
        default=50,
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
        default=['DEPREL', 'WVEC'],
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
        choices=['ADAM', 'SGD', 'ADAMW', 'LAMB'],
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


def main(args):
    global writer

    logging.basicConfig(level=logging.INFO)

    exp_name = args.solver + \
        'v4.11' + \
        '_{:1.0e}'.format(args.learning_rate) + \
        '_{:d}b'.format(args.batch_size) + \
        '_HL' + \
        '_{:d}d'.format(args.h_dims) + \
        '_{:d}l_'.format(args.h_layers) + \
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
    mark_gold_anaphores(train_raw)

    train_set = GraphDataset(
            dataset=train_raw,
            sentence_encoder=sentence_encoder,
            features=args.features
            )
    trainloader = DataLoader(
        train_set,
        batch_sampler=RandomBatchSampler(len(train_set), args.batch_size),
        num_workers=2,
        collate_fn=coref_collate
        )

    logging.info(
            'Building test graph from {}.'.format(args.test_set)
            )
    test_raw = ConlluDataset(args.test_set)
    for sentence in test_raw:
        preprocess_sentence(sentence)
    mark_gold_anaphores(test_raw)

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
        in_feats_a=75 + 5,  # mention + mention_pair
        in_feats_b=(args.h_dims + 75) * 2 + 5,
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
    writer = SummaryWriter('runs_coref/' + exp_name)

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

import time
import signal
import argparse
import logging

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_recall_fscore_support as PRF

import dgl

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence
from stroll.graph import GraphDataset
from stroll.model import MentionNet
from stroll.labels import FasttextEncoder
from stroll.loss import FocalLoss
from stroll.train import get_optimizer_and_scheduler_for_net
from stroll.train import RandomBatchSampler

MAX_MENTION_DISTANCE = 50
MAX_MENTION_PER_DOC = 1000


# Global arguments for dealing with Ctrl-C
global writer
writer = None

torch.manual_seed(43)


def train(net, trainloader,
          test_graph,
          loss_function, optimizer, scheduler,
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
        for train_graph in trainloader:
            net.train()

            xa = net(train_graph)

            target = train_graph.ndata['coref'].view(-1).clamp(0, 1)
            xa = xa.transpose(0, 1).view(2, -1)
            loss = loss_function(xa, target)

            optimizer.zero_grad()

            # for batches without a mention,  the loss is zero and
            # loss.backward() raises a RuntimeError
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                logging.error(
                        'Loss={} ignoring runtime error in torch: {}'.format(
                            loss.item(), e))

            # diagnostics
            word_count += len(train_graph)
            args.word_count = word_count
            writer.add_scalar('loss', loss.item(), word_count)

            if word_count > next_eval:
                dur = time.time() - t0
                net.eval()
                with torch.no_grad():
                    xa = net(test_graph)

                    # system mentions
                    _, system = torch.max(xa, dim=1)

                    # correct mentions:
                    target = test_graph.ndata['coref'].view(-1).clamp(0, 1)

                    # score
                    score_id_p, score_id_r, score_id_f1, _ = PRF(
                        target, system, labels=[1]
                        )

                    score_id_p = score_id_p[0]
                    score_id_r = score_id_r[0]
                    score_id_f1 = score_id_f1[0]

                # Report
                print('Elements {:08d} |'.format(word_count),
                      'F1 {:.4f}|'.format(score_id_f1),
                      'words/sec {:4.3f}'.format(count_per_eval / dur)
                      )

                writer.add_scalar('s_id_p', score_id_p, word_count)
                writer.add_scalar('s_id_r', score_id_r, word_count)
                writer.add_scalar('s_id_f1', score_id_f1, word_count)

                for name, param in net.state_dict().items():
                    writer.add_scalar(
                            'norm_' + name,
                            torch.norm(param.float()),
                            word_count
                            )

                # Save best-until-now model
                score = score_id_f1
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
    name = './runs_mentions/{}/model_{:09d}.pt'.format(
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
        num_workers=2,
        collate_fn=dgl.batch
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
        collate_fn=dgl.batch
        )

    test_graph = dgl.batch([g for g in testloader])

    logging.info('Building model.')
    net = MentionNet(
        in_feats=train_set.in_feats,
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
    writer = SummaryWriter('runs_mentions/' + exp_name)

    print('Ctrl-c will abort training and save the current model.')

    def sigterm_handler(_signo, _stack_frame):
        global writer

        writer.close()
        save_model(net)
        print('Ctrl-c detected, aborting')
        exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    loss_function = FocalLoss(
            gamma=1.5,
            alpha=None,  # FRAME_WEIGHTS,
            size_average=True
            )

    print(net)
    train(net,
          trainloader,
          test_graph,
          loss_function,
          optimizer,
          scheduler,
          args.epochs
          )

    save_model(net)
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

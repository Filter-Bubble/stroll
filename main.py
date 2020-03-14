import time
import signal
import argparse

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dgl

from stroll.graph import GraphDataset
from stroll.model import Net
from stroll.labels import BertEncoder, FasttextEncoder
from stroll.labels import FRAME_WEIGHTS, ROLE_WEIGHTS, frame_codec, role_codec
from stroll.focalloss import FocalLoss

import matplotlib.pyplot as plt
import seaborn as sns

import logging

torch.manual_seed(43)

# Not used at the moment
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f


def save_model_with_args(model, args):
    d = model.state_dict()
    d['hyperparams'] = args
    name = './runs/{}/model_{:09d}.pt'.format(args.exp_name, args.word_count)
    torch.save(d, name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
            description='Train a R-GCN for Semantic Roll Labelling.'
            )
    parser.add_argument(
            '--epochs',
            dest='epochs',
            type=int,
            default=20,
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
            dest='lr',
            type=float,
            default='1e-2',
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
            '--loss_function',
            dest='loss_function',
            default='CE',
            choices=['CE', 'FL'],
            help='Type of loss function (cross entry / focall loss)',
            )
    parser.add_argument(
            '--activation',
            dest='activation',
            default='relu',
            choices=['relu', 'tanhshrink'],
            help='Activation function for the RGCN layers.'
            )
    parser.add_argument(
            '--solver',
            dest='solver',
            default='ADAM',
            choices=['ADAM', 'SGD'],
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
    args = parser.parse_args()

    exp_name = '{}_{}_{:1.0e}_{:d}b_{:d}d_{:d}lBN_{}_{}_MLP2_{}'.format(
            args.solver,
            args.loss_function,
            args.lr,
            args.batch_size,
            args.h_dims,
            args.h_layers,
            args.activation,
            '_'.join(args.features),
            '_eBN'
            )

    if 'WVEC' in args.features:
        if args.fasttext:
            sentence_encoder = FasttextEncoder(args.fasttext)
        else:
            sentence_encoder = BertEncoder()
        exp_name += '_' + sentence_encoder.name
    else:
        sentence_encoder = None

    args.word_count = 0
    args.exp_name = exp_name
    print('Experiment {}'.format(args.exp_name))

    logging.info('Preparing test dataset.')
    train_set = GraphDataset(
            args.train_set,
            sentence_encoder=sentence_encoder,
            features=args.features
            )
    trainloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dgl.batch
        )

    logging.info('Building test graph.')
    test_set = GraphDataset(
            args.test_set,
            sentence_encoder=sentence_encoder,
            features=args.features
            )

    test_graph = dgl.batch([g for g in test_set])

    logging.info('Building model.')
    net = Net(
        in_feats=train_set.in_feats,
        h_layers=args.h_layers,
        h_dims=args.h_dims,
        out_feats_a=2,  # number of frames
        out_feats_b=21,  # number of roles
        activation=args.activation
        )
    print(net.__repr__())

    logging.info('Initializing Optimizer and learning rate scheduler.')
    if args.solver == 'SGD':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=0.9
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.9
            )
    elif args.solver == 'ADAM':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.lr
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5
            )

    logging.info('Initializing loss functions.')
    if args.loss_function == 'FL':
        focal_frame_loss = FocalLoss(
                gamma=5.,
                alpha=FRAME_WEIGHTS.view(-1),
                size_average=True
                )
        focal_role_loss = FocalLoss(
                gamma=5.,
                alpha=ROLE_WEIGHTS.view(-1),
                size_average=True
                )
    elif args.loss_function == 'CE':
        pass

    print('Looking for "restart.pt".')
    try:
        net.load_state_dict(torch.load('restart.pt'))
        print('Restart succesful.')
    except(FileNotFoundError):
        print('Restart failed, starting from scratch.')

    print('Tensorboard output in "{}".'.format(exp_name))
    writer = SummaryWriter('runs/' + exp_name)

    print('Ctr-c will abort training and save the current model.')

    def sigterm_handler(_signo, _stack_frame):
        writer.close()
        save_model_with_args(net, args)
        print('C-c detected, aborting')
        exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    # diagnostic settings
    word_count = 0
    count_per_eval = 5000
    next_eval = count_per_eval
    t0 = time.time()

    print('Start training for {:d} epochs.'.format(args.epochs))

    # loop over the epochs
    for epoch in range(args.epochs):
        writer.add_scalar(
                'learning_rate',
                optimizer.param_groups[0]['lr'],
                word_count
                )
        # loop over each minibatch
        for g in trainloader:
            net.train()

            # inputs (minibatch, C, d_1, d_2, ..., d_K)
            # target (minibatch,    d_1, d_2, ..., d_K)
            # minibatch = minibatch Size : 1
            # C         = number of classes : 2 or 21
            # d_x       = extra dimensions : number of words in graph
            logits_frame, logits_role = net(g)

            if args.loss_function == 'CE':
                target = g.ndata['frame']
                logits_frame = logits_frame.transpose(0, 1)
                loss_frame = F.cross_entropy(
                        logits_frame.view(1, 2, -1),
                        target.view(1, -1),
                        FRAME_WEIGHTS.view(1, -1))

                target = g.ndata['role']
                logits_role = logits_role.transpose(0, 1)
                loss_role = F.cross_entropy(
                        logits_role.view(1, 21, -1),
                        target.view(1, -1),
                        ROLE_WEIGHTS.view(1, -1))

            elif args.loss_function == 'FL':
                target = g.ndata['frame'].view(-1)
                logits_frame = logits_frame.transpose(0, 1)
                loss_frame = focal_frame_loss(logits_frame.view(2, -1), target)

                target = g.ndata['role'].view(-1)
                logits_role = logits_role.transpose(0, 1)
                loss_role = focal_role_loss(logits_role.view(21, -1), target)

            # add the two losses
            loss = torch.exp(-1. * net.loss_a) * loss_role + 0.5 * net.loss_a
            loss += torch.exp(-1. * net.loss_b) * loss_frame + 0.5 * net.loss_b

            # apply loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # diagnostics
            word_count = word_count + len(g)
            writer.add_scalar('wf', torch.exp(-1 * net.loss_a), word_count)
            writer.add_scalar('wr', torch.exp(-1 * net.loss_b), word_count)
            writer.add_scalar('loss_frame', loss_frame.item(), word_count)
            writer.add_scalar('loss_role', loss_role.item(), word_count)
            writer.add_scalar('loss_total', loss.item(), word_count)

            if word_count > next_eval:
                dur = time.time() - t0
                accF, accR, conf_F, conf_R = net.evaluate(test_graph)
                print('Elements {:08d} |'.format(word_count),
                      'LossF {:.4f} |'.format(loss_role.item()),
                      'LossR {:.4f} |'.format(loss_frame.item()),
                      'AccF {:.4f} |'.format(accF),
                      'AccR {:.4f} |'.format(accR),
                      'words/sec {:4.3f}'.format(len(g) / dur)
                      )
                figure = plt.figure(figsize=[10., 10.])
                labels = frame_codec.classes_
                fmt = ".0f"
                sns.heatmap(
                        conf_F, fmt=fmt, annot=True, cbar=False, cmap="Greens",
                        xticklabels=labels, yticklabels=labels
                        )
                writer.add_figure('confusion_matrix-Frame', figure, word_count)

                figure = plt.figure(figsize=[10., 10.])
                labels = role_codec.classes_
                sns.heatmap(
                        conf_R, fmt=fmt, annot=True, cbar=False,
                        cmap="Greens", xticklabels=labels, yticklabels=labels
                        )
                writer.add_figure('confusion_matrix-Role', figure, word_count)

                writer.add_scalar('accuracy_role', accR, word_count)
                writer.add_scalar('accuracy_frame', accF, word_count)

                for name, param in net.state_dict().items():
                    writer.add_histogram('hist_' + name, param, word_count)
                    if param.dtype == torch.int64:
                        writer.add_scalar(
                                'norm_' + name,
                                torch.norm(param.float()),
                                word_count
                                )
                    else:
                        writer.add_scalar(
                                'norm_' + name,
                                torch.norm(param),
                                word_count
                                )

                # reset timer
                next_eval = next_eval + count_per_eval
                t0 = time.time()

        args.word_count = word_count
        save_model_with_args(net, args)
        print('Epoch {} done'.format(epoch))

        scheduler.step()
        writer.add_scalar(
                'learning_rate',
                optimizer.param_groups[0]['lr'],
                word_count-1
                )

    save_model_with_args(net, args)
    writer.close()

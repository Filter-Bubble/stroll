import sys
import time
import signal
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import dgl

from stroll.graph import GraphDataset
from stroll.model import Net
from stroll.labels import FRAME_WEIGHTS, ROLE_WEIGHTS, \
        ROLE_TARGET_DISTRIBUTIONS
from stroll.labels import BertEncoder, FasttextEncoder, \
        role_codec, frame_codec
from stroll.loss import CrossEntropy, FocalLoss, Bhattacharyya, \
        HingeSquared, KullbackLeibler

import matplotlib.pyplot as plt
import seaborn as sns


# Global arguments for dealing with Ctrl-C
writer = None
args = None

torch.manual_seed(43)

# Not used at the moment
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f


def get_loss_functions(loss_function='CE', gamma=1.5):
    if loss_function == 'FL':
        frame_loss = FocalLoss(
                gamma=gamma,
                alpha=None,  # FRAME_WEIGHTS,
                size_average=True
                )
        role_loss = FocalLoss(
                gamma=gamma,
                alpha=None,  # ROLE_WEIGHTS,
                size_average=True
                )

    elif loss_function == 'FB':
        frame_loss = FocalLoss(
                gamma=gamma,
                alpha=None,  # FRAME_WEIGHTS,
                size_average=True
                )
        role_loss = Bhattacharyya(
                target_distribution=ROLE_TARGET_DISTRIBUTIONS
                )

    elif loss_function == 'FK':
        frame_loss = FocalLoss(
                gamma=gamma,
                alpha=None,  # FRAME_WEIGHTS,
                size_average=True
                )
        role_loss = KullbackLeibler(
                target_distribution=ROLE_TARGET_DISTRIBUTIONS
                )

    elif loss_function == 'FH':
        frame_loss = FocalLoss(
                gamma=gamma,
                alpha=None,  # FRAME_WEIGHTS,
                size_average=True
                )
        role_loss = HingeSquared()

    elif loss_function == 'CE':
        frame_loss = CrossEntropy(
                classes=len(frame_codec.classes_),
                weights=FRAME_WEIGHTS.view(1, -1)
                )
        role_loss = CrossEntropy(
                classes=len(role_codec.classes_),
                weights=ROLE_WEIGHTS.view(1, -1)
                )
    else:
        print('Loss function not implemented.')
        sys.exit(-1)

    return frame_loss, role_loss


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


def evaluate(net, g):
    net.eval()
    with torch.no_grad():
        frame_labels, role_labels, \
                frame_chance, role_chance = net.label(g)

        correct_frame = frame_codec.inverse_transform(g.ndata['frame'])
        correct_role = role_codec.inverse_transform(g.ndata['role'])

        acc_F = f1_score(correct_frame, frame_labels,
                         average='macro', zero_division=0)
        acc_R = f1_score(correct_role, role_labels,
                         average='macro', zero_division=0)

        normalize = 'true'  # 'true': normalize wrt. the true label count
        conf_F = 100. * confusion_matrix(
                correct_frame, frame_labels,
                normalize=normalize
                )
        conf_R = 100. * confusion_matrix(
                correct_role, role_labels,
                normalize=normalize
                )

        return acc_F, acc_R, conf_F, conf_R


def train(net, trainloader, test_graph,
          combine_loss='cst',
          epochs=60,
          ):
    # diagnostic settings
    count_per_eval = 5000
    next_eval = count_per_eval
    t0 = time.time()
    best_model_accuracy = 0.
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
        for g in trainloader:
            net.train()

            # inputs (minibatch, C, d_1, d_2, ..., d_K)
            # target (minibatch,    d_1, d_2, ..., d_K)
            # minibatch = minibatch Size : 1
            # C         = number of classes : 2 or 19
            # d_x       = extra dimensions : number of words in graph
            logits_frame, logits_role = net(g)

            target = g.ndata['frame'].view(-1)
            logits_frame = logits_frame.transpose(0, 1)
            loss_frame = frame_loss(logits_frame.view(2, -1), target)

            target = g.ndata['role'].view(-1)
            logits_role = logits_role.transpose(0, 1)
            loss_role = role_loss(logits_role.view(19, -1), target)

            # add the two losses
            if combine_loss == 'dyn':
                total_loss = torch.exp(-1. * net.loss_a) * loss_role + \
                    0.5 * net.loss_a
                total_loss += torch.exp(-1. * net.loss_b) * loss_frame + \
                    0.5 * net.loss_b
            elif combine_loss == 'cst':
                total_loss = loss_role + loss_frame
            else:
                sys.exit(-1)

            # apply loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # diagnostics
            word_count += len(g)
            args.word_count = word_count
            writer.add_scalar('loss_frame', loss_frame.item(), word_count)
            writer.add_scalar('loss_role', loss_role.item(), word_count)
            writer.add_scalar('loss_total', total_loss.item(), word_count)

            if word_count > next_eval:
                dur = time.time() - t0
                accF, accR, conf_F, conf_R = evaluate(net, test_graph)
                print('Elements {:08d} |'.format(word_count),
                      'AccF {:.4f} |'.format(accF),
                      'AccR {:.4f} |'.format(accR),
                      'words/sec {:4.3f}'.format(len(g) / dur)
                      )

                figure = plt.figure(figsize=[10., 10.])
                labels = role_codec.classes_
                sns.heatmap(
                        conf_R, fmt='.0f', annot=True, cbar=False,
                        cmap="Greens", xticklabels=labels, yticklabels=labels
                        )
                writer.add_figure('confusion_matrix-Role', figure, word_count)

                writer.add_scalar('accuracy_role', accR, word_count)
                writer.add_scalar('accuracy_frame', accF, word_count)

                for name, param in net.state_dict().items():
                    writer.add_scalar(
                            'norm_' + name,
                            torch.norm(param.float()),
                            word_count
                            )

                # Save best-until-now model
                if epoch > 0 and accR > best_model_accuracy:
                    logging.info('Saving new best model at step {:09d}'.format(
                        word_count
                        ))
                    best_model_accuracy = accR
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
        description='Train a R-GCN for Semantic Roll Labelling.'
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
        default='1e-2',
        help='Initial learning rate.'
        )
parser.add_argument(
        '--features',
        nargs='*',
        dest='features',
        default=['UPOS', 'FEATS', 'DEPREL', 'WVEC'],
        choices=['UPOS', 'XPOS', 'FEATS', 'DEPREL', 'WVEC', 'RID'],
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
        choices=['CE', 'FL', 'FB', 'FK', 'FH'],
        help='Type of loss function',
        )
parser.add_argument(
        '--loss_gamma',
        dest='loss_gamma',
        type=float,
        default=5,
        help='Exponent gamma in FocalLoss'
        )
parser.add_argument(
        '--combine_loss',
        dest='combine_loss',
        default='cst',
        choices=['cst', 'dyn'],
        help='How to combine the two losses.'
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    exp_name = args.solver + \
        '_{:1.0e}'.format(args.learning_rate) + \
        '_{}{:1.2e}'.format(args.loss_function, args.loss_gamma) + \
        args.combine_loss + \
        '_{:d}b'.format(args.batch_size) + \
        '_{:d}d'.format(args.h_dims) + \
        '_{:d}l'.format(args.h_layers) + \
        '_' + args.activation + \
        '_'.join(args.features)

    if 'WVEC' in args.features:
        if args.fasttext:
            sentence_encoder = FasttextEncoder(args.fasttext)
        else:
            sentence_encoder = BertEncoder()
        exp_name += '_' + sentence_encoder.name
    else:
        sentence_encoder = None

    args.exp_name = exp_name
    print('Experiment {}'.format(args.exp_name))

    logging.info(
            'Preparing train dataset {}'.format(args.train_set)
            )
    train_set = GraphDataset(
            args.train_set,
            sentence_encoder=sentence_encoder,
            features=args.features
            )
    trainloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=dgl.batch
        )

    logging.info(
            'Building test graph from {}.'.format(args.test_set)
            )
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
        out_feats_b=19,  # number of roles
        activation=args.activation
        )
    logging.info(net.__repr__())

    logging.info('Initializing Optimizer and learning rate scheduler.')
    optimizer, scheduler = get_optimizer_and_scheduler_for_net(
            net,
            args.solver,
            args.learning_rate
            )

    logging.info('Initializing loss functions.')
    frame_loss, role_loss = get_loss_functions(
            loss_function=args.loss_function,
            gamma=args.loss_gamma
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
        writer.close()
        save_model(net)
        print('Ctrl-c detected, aborting')
        exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    train(net,
          trainloader,
          test_graph,
          args.combine_loss,
          args.epochs
          )

    save_model(net)
    writer.close()

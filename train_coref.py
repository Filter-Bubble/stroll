import sys
import time
import signal
import argparse
import logging

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stroll.loss import FocalLoss

from sklearn.metrics import precision_recall_fscore_support as PRF
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AffinityPropagation

import dgl

from stroll.graph import GraphDataset
from stroll.model import Net
from stroll.labels import FasttextEncoder


# Global arguments for dealing with Ctrl-C
writer = None
args = None

torch.manual_seed(43)


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


def train(net, trainloader, test_graph, epochs=60):
    # diagnostic settings
    count_per_eval = 5000
    next_eval = count_per_eval
    t0 = time.time()
    best_model_score = 0.
    word_count = args.word_count

    id_loss_f = FocalLoss(
            gamma=1.5,
            alpha=None,  # unweighted
            size_average=True
            )

    # estimator = AffinityPropagation(
    #         affinity='precomputed',
    #         preference=-50,
    #         damping=0.66)

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

            # prediceted labels
            id_out, match_out = net(g)
            id_out = id_out.transpose(0, 1)

            # correct labels
            # == -1 : token is not a mention -> 0
            # >=  0 : token is a mention     -> 1
            target = g.ndata['coref'].view(-1).clamp(-1, 0) + 1

            # loss
            loss_id = id_loss_f(id_out.view(2, -1), target)

            # actual distance in sentences, as a 2D tensor
            sent_idx = g.ndata['index'].view(-1).unsqueeze(1).float()
            sent_dist = torch.pdist(sent_idx, p=1.0)
            writer.add_scalar('sent_dist', sent_dist.mean().item(), word_count)

            # predicted distances, from the scipy documention:
            # pdist returns a condensed distance matrix Y.
            # For each i and j (where i < j < m), where m is the number of
            # original observations, the metric dist(u=X[i], v=X[j]) is
            # computed and stored in entry ij.
            pdist = torch.pdist(match_out, p=2.0)

            #    torch.exp(
            #            net.loss_a * sent_dist + net.loss_b
            #            )
            writer.add_scalar('pdist', pdist.mean().item(), word_count)

            # gold similarities
            #   i         j      objective
            # corefA x corefA    equal, minimize dist
            # corefA x corefB    different, dist > 1
            # coref  x   _       different, dist > 1
            #   _    x   _       different, dist > 1

            mentions = g.ndata['coref'].view(-1)
            target_min_set = []
            target_max_set = []

            for m1 in range(len(g)):
                for m2 in range(m1):
                    m1m2 = len(g)*m2 - m2*(m2+1)/2 + m1 - 1 - m2
                    # only minimize between mentions
                    # and maximize between mentions and non-mentions
                    if mentions[m2] == mentions[m1] and mentions[m1] > -0.5:
                        target_min_set.append(m1m2)
                    elif mentions[m1] > -0.5 or mentions[m2] > -0.5:
                        target_max_set.append(m1m2)

            d1 = torch.exp(net.loss_a)
            loss_min_dist = torch.gather(
                    pdist * d1, 0,
                    torch.tensor(target_min_set, dtype=torch.int64)
                    ).mean()

            d2 = torch.exp(net.loss_b)
            max_dist = torch.gather(
                    pdist, 0,
                    torch.tensor(target_max_set, dtype=torch.int64)
                    )
            max_dist = 5.0 * d2 - max_dist  # apply margin
            max_dist[max_dist < 0] = 0  # hinge
            max_dist = max_dist**2  # squared
            loss_max_dist = max_dist.mean()

            total_loss = loss_id + loss_min_dist + loss_max_dist + \
                net.loss_a**2 + net.loss_b**2
            # total_loss = loss_min_dist + loss_max_dist

            # apply loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # diagnostics
            word_count += len(g)
            args.word_count = word_count
            writer.add_scalar('loss_id', loss_id.item(), word_count)
            writer.add_scalar('loss_min', loss_min_dist.item(), word_count)
            writer.add_scalar('loss_max', loss_max_dist.item(), word_count)
            writer.add_scalar('loss_a', net.loss_a.item(), word_count)
            writer.add_scalar('loss_b', net.loss_b.item(), word_count)
            writer.add_scalar('loss_total', total_loss.item(), word_count)

            if word_count > next_eval:
                dur = time.time() - t0
                net.eval()
                with torch.no_grad():
                    id_out, match_out = net(test_graph)
                    _, system = torch.max(id_out, dim=1)

                    correct = test_graph.ndata['coref'].view(-1)
                    correct = correct.clamp(-1, 0) + 1

                    score_p, score_r, score_f, _ = PRF(
                        correct, system, labels=[1]
                        )

                    score_p = score_p[0]
                    score_r = score_r[0]
                    score_f = score_f[0]

                    # super slow...
                    # # get all gold coref annotations
                    # coref = test_graph.ndata['coref'].view(-1)

                    # # pick the actual mentions
                    # correct_idx = np.where(coref >= 0)

                    # # get the correct clusters
                    # correct = coref[correct_idx]
                    # nmentions = len(correct)

                    # # get the pairwise distances for the system's mentions,
                    # # assuming gold mentions
                    # match_out = match_out[correct_idx]
                    # pdist = torch.pdist(match_out)

                    # # build affinity matrix
                    # affinities = np.zeros([
                    #     nmentions, nmentions])

                    # for m1 in range(nmentions):
                    #     affinities[m1, m1] = 1.0
                    #     for m2 in range(m1):
                    #         m1m2 = nmentions*m2 - m2*(m2+1)/2 + m1 - 1 - m2
                    #         m1m2 = int(m1m2)
                    #         affinities[m1, m2] = - pdist[m1m2]
                    #         affinities[m2, m1] = - pdist[m1m2]

                    # estimator.fit(affinities)
                    # score = adjusted_rand_score(correct, estimator.labels_)
                    score = 0

                print('Elements {:08d} |'.format(word_count),
                      'P {:.4f} |'.format(score_p),
                      'R {:.4f} |'.format(score_r),
                      'F1 {:.4f}|'.format(score_f),
                      'AR {:.4f}|'.format(score),
                      'words/sec {:4.3f}'.format(len(g) / dur)
                      )

                writer.add_scalar('p', score_p, word_count)
                writer.add_scalar('r', score_r, word_count)
                writer.add_scalar('f1', score_f, word_count)
                writer.add_scalar('ar', score, word_count)

                for name, param in net.state_dict().items():
                    writer.add_scalar(
                            'norm_' + name,
                            torch.norm(param.float()),
                            word_count
                            )

                # Save best-until-now model
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
        choices=['UPOS', 'XPOS', 'FEATS', 'DEPREL', 'WVEC', 'RID', 'IDX'],
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    exp_name = args.solver + \
        'mm_normalized' + \
        '_{:1.0e}'.format(args.learning_rate) + \
        '_{:d}b'.format(args.batch_size) + \
        '_FL1.5' + \
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
    train_set = GraphDataset(
            args.train_set,
            sentence_encoder=sentence_encoder,
            features=args.features
            )
    trainloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
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
        out_feats_a=2,  # number mention or non-mention
        out_feats_b=128,  # number of roles
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
          args.epochs
          )

    save_model(net)
    writer.close()

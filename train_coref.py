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

import dgl

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence, build_mentions_from_heads
from stroll.coref import features_mention, features_mention_pair
from stroll.graph import GraphDataset
from stroll.model import CorefNet
from stroll.labels import FasttextEncoder

MAX_MENTION_DISTANCE = 30


# Global arguments for dealing with Ctrl-C
global writer
writer = None

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


def build_pairs(raw, graph, idxs, wvec_out):
    wvec = wvec_out[idxs]
    entity = graph.ndata['coref'][idxs]

    sent_index = graph.ndata['index']
    token_index = graph.ndata['token_index']

    mentions = []
    for idx in idxs:
        sentence = raw[sent_index[idx].item()]
        token = sentence[token_index[idx].item()]
        mentions += build_mentions_from_heads(
                sentence, [token.ID]
                )

    # build pairs
    target = []
    vectors = []
    for mid, mention in enumerate(mentions):
        for aid in range(max(0, mid - MAX_MENTION_DISTANCE), mid):
            antecedent = mentions[aid]
            # only match within a document
            if antecedent.sentence.doc_id == mention.sentence.doc_id:

                if entity[aid] == entity[mid]:
                    # NOTE: this includes not entity with non-entity,
                    # so coref == -1 for both.
                    target.append(1.0)
                else:
                    target.append(0.0)

                # build pair (aid, mid)
                vectors.append(
                    torch.cat((
                        wvec[mid].view(-1),
                        features_mention(raw, mention),
                        wvec[aid].view(-1),
                        features_mention(raw, antecedent),
                        features_mention_pair(
                            raw,
                            antecedent,
                            mention)
                        ))
                )
    vectors = torch.stack(vectors)
    return vectors, target


def train(net, train_raw, trainloader, test_raw, test_graph,
          optimizer, scheduler,
          epochs=60):
    global writer

    # diagnostic settings
    count_per_eval = 25000
    next_eval = count_per_eval
    t0 = time.time()
    best_model_score = 0.
    word_count = args.word_count

    id_loss_f = FocalLoss(
            gamma=1.5,
            alpha=None,  # unweighted
            size_average=True
            )

    sim_loss_f = FocalLoss(
            gamma=1.5,
            alpha=None,  # unweighted
            size_average=True
            )

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

            # predict mentions, and vectors
            id_out, wvec_out = net(train_graph)

            # correct mentions
            # == -1 : token is not a mention -> 0
            # >=  0 : token is a mention     -> 1
            target = train_graph.ndata['coref'].view(-1).clamp(-1, 0) + 1

            # score mentions
            loss_id = id_loss_f(id_out.transpose(0, 1).view(2, -1), target)

            # predict coreference pairs:

            # take the indices of the nodes that are mentions
            _, mention_idxs = torch.max(id_out, dim=1)
            mention_idxs = torch.nonzero(mention_idxs)

            if len(mention_idxs) != 0:
                pairvecs, target = build_pairs(
                        train_raw,
                        train_graph,
                        mention_idxs,
                        wvec_out
                        )
                similarities = net.task_b(pairvecs)

                # correct pairs
                target = torch.tensor(target)

                # score pairs
                loss_sim = sim_loss_f(
                        similarities.transpose(0, 1).view(2, -1),
                        target
                        )
            else:
                loss_sim = 1.0

            # apply loss
            loss_total = MAX_MENTION_DISTANCE * loss_id + loss_sim
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # diagnostics
            word_count += len(train_graph)
            args.word_count = word_count
            writer.add_scalar('loss_id', loss_id.item(), word_count)
            writer.add_scalar('loss_sim', loss_sim.item(), word_count)
            writer.add_scalar('loss_total', loss_total.item(), word_count)

            if word_count > next_eval:
                dur = time.time() - t0
                net.eval()
                with torch.no_grad():
                    id_out, wvec_out = net(test_graph)
                    _, system = torch.max(id_out, dim=1)

                    correct = test_graph.ndata['coref'].view(-1)
                    correct = correct.clamp(-1, 0) + 1

                    score_id_p, score_id_r, score_id_f, _ = PRF(
                        correct, system, labels=[1]
                        )

                    score_id_p = score_id_p[0]
                    score_id_r = score_id_r[0]
                    score_id_f = score_id_f[0]

                    # predict coreference pairs
                    _, mention_idxs = torch.max(id_out, dim=1)
                    mention_idxs = torch.nonzero(mention_idxs)
                    pairvecs, correct = build_pairs(
                            test_raw,
                            test_graph,
                            mention_idxs,
                            wvec_out
                            )
                    system = net.task_b(pairvecs)
                    _, system = torch.max(system, dim=1)

                    score_sim_p, score_sim_r, score_sim_f, _ = PRF(
                        correct, system, labels=[1]
                        )

                    score_sim_p = score_sim_p[0]
                    score_sim_r = score_sim_r[0]
                    score_sim_f = score_sim_f[0]

                print('Elements {:08d} |'.format(word_count),
                      'F1 {:.4f}|'.format(score_id_f),
                      'F1 {:.4f}|'.format(score_sim_f),
                      'words/sec {:4.3f}'.format(len(train_graph) / dur)
                      )

                writer.add_scalar('s_id_p', score_id_p, word_count)
                writer.add_scalar('s_id_r', score_id_r, word_count)
                writer.add_scalar('s_id_f1', score_id_f, word_count)
                writer.add_scalar('s_sim_p', score_sim_p, word_count)
                writer.add_scalar('s_sim_r', score_sim_r, word_count)
                writer.add_scalar('s_sim_f1', score_sim_f, word_count)

                for name, param in net.state_dict().items():
                    writer.add_scalar(
                            'norm_' + name,
                            torch.norm(param.float()),
                            word_count
                            )

                # Save best-until-now model
                if epoch > 0 and score_id_f * score_sim_f > best_model_score:
                    logging.info('Saving new best model at step {:09d}'.format(
                        word_count
                        ))
                    best_model_score = score_id_f * score_sim_f
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
        'v3.3' + \
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
        batch_size=args.batch_size,
        # num_workers=2,
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
    test_graph = dgl.batch([g for g in test_set])

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
          train_raw,
          trainloader,
          test_raw,
          test_graph,
          optimizer,
          scheduler,
          args.epochs
          )

    save_model(net)
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

import argparse
import logging
import numpy as np

import torch

from torch.utils.data import DataLoader
from scorch.scores import muc, b_cubed, ceaf_m, ceaf_e, blanc

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import precision_recall_fscore_support as PRF
from stroll.evaluate import variation_of_information

import dgl

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence
from stroll.coref import get_mentions
from stroll.coref import mark_gold_anaphores
from stroll.coref import nearest_linking
from stroll.coref import predict_anaphores, predict_similarities
from stroll.graph import GraphDataset
from stroll.model import CorefNet
from stroll.labels import FasttextEncoder
from stroll.evaluate import clusters_to_sets

MAX_MENTION_PER_DOC = 1000

parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument(
        '--model',
        dest='model_name',
        help='Model to evaluate',
        required=True
        )
parser.add_argument(
        'dataset',
        help='Evaluation dataset in conllu format',
        )


def coref_collate(batch):
    """
    Collate function to batch samples together.
    """

    mentions = []
    for g in batch:
        mentions += get_mentions(g.sentence)
    return dgl.batch(batch), mentions


def evaluate(net, test_graph, test_mentions, test_clusters, namenpz):
    net.eval()
    with torch.no_grad():
        gvec = net(test_graph)

        # predict links
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

        # Prepare anaphore scoring
        targets = torch.tensor([
            mention.anaphore for mention in test_mentions
            ])

        # Prepare the cluster scoring
        gold_sets = clusters_to_sets(test_clusters)

        # * header of printed table
        columns = [
                'margin',
                'Pana', 'Rana', 'Fana', 'Sana',
                'ARI', 'AMI',
                'MUCR', 'MUCP', 'MUCF',
                'B3R', 'B3P', 'B3F',
                'CEAFmR', 'CEAFmP', 'CEAFmF',
                'CEAFeR', 'CEAFeP', 'CEAFeF',
                'BLANCR', 'BLANCP', 'BLANCF',
                'CoNLL', 'VI'
                ]
        print('	'.join(columns))

        for margin in [-0.25, -0.20, -0.15, -0.10, -0.05,
                       0.0,
                       0.05, 0.10, 0.15, 0.20, 0.15]:
            # score the anaphores using the margin
            ana_scores = PRF(
                    targets,
                    torch.round(anaphores - margin),
                    labels=[1.0]
                    )

            # cluster at this margin
            system_clusters = nearest_linking(
                    similarities, anaphores, margin=margin
                    )
            system_sets = clusters_to_sets(system_clusters)

            mucs = muc(gold_sets, system_sets)
            b3s = b_cubed(gold_sets, system_sets)
            cms = ceaf_m(gold_sets, system_sets)
            ces = ceaf_e(gold_sets, system_sets)
            bls = blanc(gold_sets, system_sets)
            vi = variation_of_information(test_clusters, system_clusters)

            ari = adjusted_rand_score(
                    test_clusters, system_clusters
                    )
            ami = adjusted_mutual_info_score(
                    test_clusters, system_clusters
                    )
            conll = 0.333 * (b3s[2] + mucs[2] + ces[2])

            print(('{}' + '	{}'*(len(columns)-1)).format(
                margin,
                ana_scores[0][0], ana_scores[1][0],
                ana_scores[2][0], ana_scores[3][0],
                ari, ami,
                mucs[0], mucs[1], mucs[2],
                b3s[0], b3s[1], b3s[2],
                cms[0], cms[1], cms[2],
                ces[0], ces[1], ces[2],
                bls[0], bls[1], bls[2],
                conll, vi
                ))

        np.savez(
                namenpz,
                similarities=similarities, links=links,
                test_clusters=test_clusters,
                system_clusters=system_clusters,
                anaphores=anaphores.numpy()
                )


if __name__ == '__main__':
    args = parser.parse_args()

    state_dict = torch.load(args.model_name)
    hyperparams = state_dict.pop('hyperparams')

    if 'WVEC' in hyperparams.features:
        if hyperparams.fasttext:
            sentence_encoder = FasttextEncoder(hyperparams.fasttext)
    else:
        sentence_encoder = None

    logging.info(
            'Building eval graph from {}.'.format(args.dataset)
            )
    test_conllu = ConlluDataset(args.dataset)
    for sentence in test_conllu:
        preprocess_sentence(sentence)
    mark_gold_anaphores(test_conllu)

    test_set = GraphDataset(
            dataset=test_conllu,
            sentence_encoder=sentence_encoder,
            features=hyperparams.features
            )
    test_loader = DataLoader(
        test_set,
        num_workers=2,
        batch_size=20,
        collate_fn=coref_collate
        )

    test_graph = []
    test_clusters = []

    test_mentions = []
    for g, m in test_loader:
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
        in_feats=test_set.in_feats,
        in_feats_a=75 + 5,
        in_feats_b=(hyperparams.h_dims + 75) * 2 + 5,
        h_layers=hyperparams.h_layers,
        h_dims=hyperparams.h_dims,
        activation='relu'
        )
    logging.info(net.__repr__())

    net.load_state_dict(state_dict, strict=False)

    # .../model_001280889.pt
    namenpz = 'test_' + args.model_name[-12:-3] + '.npz'
    evaluate(net, test_graph, test_mentions, test_clusters, namenpz)

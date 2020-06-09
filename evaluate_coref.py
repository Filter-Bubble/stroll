import argparse
import logging
import numpy as np

import torch

from torch.utils.data import DataLoader
from scorch.scores import muc, b_cubed, ceaf_m, ceaf_e, blanc

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.cluster import contingency_matrix

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


def mentions_can_link(aid, mid, antecedent, mention):
    if mid - aid >= MAX_MENTION_DISTANCE:
        # TODO: fill with exponentially decaying similarity?
        return False

    if antecedent.sentence.doc_rank != mention.sentence.doc_rank:
        # TODO: fill with very low similarities?
        return False

    return True


def predict_clusters(similarities, nlinks=200, word_count=0):
    # take the upper diagonal as needed for linkage
    fulldist = similarities.numpy()
    fulldist = fulldist[np.triu_indices(len(fulldist), 1)]

    # turn similarities into distances
    fulldist = np.nan_to_num(np.exp(-1. * fulldist))

    Z = linkage(fulldist, 'single')
    cutoff = Z[nlinks, 2]

    return list(fcluster(Z, cutoff, criterion='distance'))


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
    similarities = torch.ones([nmentions, nmentions]) * -1e8

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


def variation_of_information(clustA, clustB):
    c = contingency_matrix(clustA, clustB)
    pa = c.sum(axis=1)
    pb = c.sum(axis=0)
    logc = np.log(np.where(c < 1, 1, c))
    logpa = np.broadcast_to(np.log(pa), (len(pb), len(pa))).transpose()
    logpb = np.broadcast_to(np.log(pb), (len(pa), len(pb)))

    return -np.sum(c * (2 * logc - logpa - logpb))


def evaluate(net, eval_graph, eval_mentions, eval_clusters):
    net.eval()
    with torch.no_grad():
        gvec = net(eval_graph)

        # coreference pairs: score clustering on gold mentions

        # correct mentions:
        target = eval_graph.ndata['coref'].view(-1).clamp(0, 1)
        mention_idxs = torch.nonzero(target)

        links, similarities = predict_similarities(
                net,
                eval_mentions,
                gvec[mention_idxs]
                )

        np.savez(
                'similarities.npz',
                similarities=similarities, links=links,
                eval_clusters=eval_clusters
                )

        hashed_clusters = {}
        for i, c in enumerate(eval_clusters):
            if c in hashed_clusters:
                hashed_clusters[c].add(i)
            else:
                hashed_clusters[c] = set()
                hashed_clusters[c].add(i)
        gold_set = hashed_clusters.values()
        print('Number of system clusters: {}'.format(len(gold_set)))

        columns = [
                'nlinks', 'ARI', 'AMI', 'MUCR', 'MUCP', 'MUCF', 'B3R', 'B3P',
                'B3F', 'CEAFmR', 'CEAFmP', 'CEAFmF', 'CEAFeR', 'CEAFeP',
                'CEAFeF', 'BLANCR', 'BLANCP', 'BLANCFV', 'CoNLL', 'VI'
                ]
        print('	'.join(columns))
        for nlinks in range(20, len(mention_idxs), 20):
            system_clusters = predict_clusters(
                    similarities,
                    nlinks=nlinks
                    )

            hashed_clusters = {}
            for i, c in enumerate(system_clusters):
                if c in hashed_clusters:
                    hashed_clusters[c].add(i)
                else:
                    hashed_clusters[c] = set()
                    hashed_clusters[c].add(i)
            system_set = hashed_clusters.values()

            mucs = muc(gold_set, system_set)
            b3s = b_cubed(gold_set, system_set)
            cms = ceaf_m(gold_set, system_set)
            ces = ceaf_e(gold_set, system_set)
            bls = blanc(gold_set, system_set)
            vi = variation_of_information(eval_clusters, system_clusters)

            ari = adjusted_rand_score(
                    eval_clusters, system_clusters
                    )
            ami = adjusted_mutual_info_score(
                    eval_clusters, system_clusters
                    )
            conll = 0.3 * (b3s[2] + mucs[2] * cms[2])

            print(('{}' + '	{}'*19).format(
                nlinks, ari, ami,
                mucs[0], mucs[1], mucs[2],
                b3s[0], b3s[1], b3s[2],
                cms[0], cms[1], cms[2],
                ces[0], ces[1], ces[2],
                bls[0], bls[1], bls[2],
                conll, vi
                ))


def nearest_linking(similarity):
    entities = []

    nmentions = len(similarity)
    for i in range(1, nmentions):
        # take the similarity to all possible antecendents
        antecedent_sims = similarity[0:i, i]

        # find the most similar
        antecedent = np.argmax(antecedent_sims)

        # link if allowed
        if similarity[antecedent, i] > 0:
            # find the set that contains antecedent
            for entity in entities:
                if antecedent in entity:
                    entity.add(i)
                    break
        else:
            # start a new entity
            entities.append(set([i]))

    clusters = np.zeros(nmentions)
    for e, entity in enumerate(entities):
        for i in entity:
            clusters[i] = e

    return clusters


def parse_z(Z, similarity, mentions):
    from itertools import product
    res = ""

    nmentions = len(Z) + 1

    clusters = {}

    # starting point: each element its own cluster
    for i in range(nmentions):
        clusters[i] = set([i])

    # Z[i, 0] and Z[i, 1] are combined to form cluster n+i.
    # The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]
    # Z[i, 3] represents the number of original observations in the new cluster
    for i in range(len(Z)):
        clustA = int(Z[i, 0])
        clustB = int(Z[i, 1])
        clusters[len(clusters)] = set(
                list(clusters[clustA]) + list(clusters[clustB])
                )

        mindist = 1000
        for a, b in product(clusters[clustA], clusters[clustB]):
            if similarity[a, b] <= mindist:
                pair = (a, b)

        res += 'Merging {} ({} + {}) due to pair {}\n'.format(
                i, list(clusters[clustA]), list(clusters[clustB]), pair
                )
        m0 = mentions[pair[0]]
        m1 = mentions[pair[1]]
        res += m0.__repr__() + '\n'
        res += '{}'.format([m0.sentence[i].FORM for i in m0.ids])
        res += '\n'
        res += m1.__repr__() + '\n'
        res += '{}'.format([m1.sentence[i].FORM for i in m1.ids])
        res += '\n'
        res += '\n'
        for m in list(clusters[clustA]):
            m0 = clusters[m]
            res += '{} {} {}'.format(m, m0.refid, [m0.sentence[i].FORM for i in m0.ids])
        res += '---'
        for m in list(clusters[clustB]):
            m0 = clusters[m]
            res += '{} {} {}'.format(m, m0.refid, [m0.sentence[i].FORM for i in m0.ids])

    return res


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
    eval_conllu = ConlluDataset(args.dataset)
    for sentence in eval_conllu:
        preprocess_sentence(sentence)

    eval_set = GraphDataset(
            dataset=eval_conllu,
            sentence_encoder=sentence_encoder,
            features=hyperparams.features
            )
    evalloader = DataLoader(
        eval_set,
        num_workers=2,
        batch_size=20,
        collate_fn=coref_collate
        )

    eval_graph = []
    eval_clusters = []

    eval_mentions = []
    for g, m in evalloader:
        eval_graph.append(g)
        eval_mentions += m

    eval_graph = dgl.batch(eval_graph)
    for mention in eval_mentions:
        sentence = mention.sentence
        eval_clusters.append(
                int(mention.refid) + sentence.doc_rank * MAX_MENTION_PER_DOC
                )

    logging.info('Building model.')
    net = CorefNet(
        in_feats=eval_set.in_feats,
        in_feats_b=(hyperparams.h_dims + 7) * 2 + 5,
        h_layers=hyperparams.h_layers,
        h_dims=hyperparams.h_dims,
        activation='relu'
        )
    logging.info(net.__repr__())

    net.load_state_dict(state_dict, strict=False)

    evaluate(net, eval_graph, eval_mentions, eval_clusters)

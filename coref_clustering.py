import sys
import argparse
import logging

import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_rand_score

from scipy.optimize import minimize

import matplotlib.pyplot as plt

from stroll.graph import ConlluDataset
from stroll.coref import preprocess_sentence, postprocess_sentence

parser = argparse.ArgumentParser(
        description='Create coreference chains from a similarity matrix',
        )
parser.add_argument(
        '--conll',
        help='Annotated conll dataset'
        )


def build_scoring_file(dataset, docname, filename):
    keyfile = open(filename, 'w')
    keyfile.write('#begin document ({});\n'.format(args.conll))
    isFirst = True
    for sentence in dataset:
        if isFirst:
            isFirst = False
        else:
            keyfile.write('\n')
        for token in sentence:
            if token.FORM == '':
                # these are from unfolding the coordination clauses, dont print
                continue
            if token.COREF != '_':
                coref = token.COREF
            else:
                coref = '-'
            keyfile.write('{}\t0\t{}\t{}\t{}\n'.format(
                sentence.sent_id, token.ID, token.FORM, coref))

    keyfile.write('#end document\n')
    keyfile.close()


def cluster_mentions(dataset):
    mentions = []
    # 1. find all mentions
    for sentence in dataset:
        for token in sentence:
            if token.COREF != '_':
                mentions.append(token.COREF)

    # 2a. make similarity matrix
    similarity = np.zeros([len(mentions), len(mentions)])
    for m1, entity in enumerate(mentions):
        similarity[m1, m1] = 1.0
        for m2 in range(m1):
            if mentions[m2] == mentions[m1]:
                similarity[m1, m2] = -1.0
                similarity[m2, m1] = -1.0
            else:
                similarity[m1, m2] = -20.0
                similarity[m2, m1] = -20.0

    similarity += np.random.normal(scale=5, size=len(mentions)**2).reshape(
            [len(mentions), len(mentions)]
            )

    # 2b. add noise?

    # 3. plot it
    plt.imshow(similarity, cmap='hot', interpolation='nearest')
    plt.show()

    # 5. try clustering approaches
    best_params = np.zeros([2])
    best_score = 0
    for preference in np.arange(-100, -1, 5):
        for damping in np.arange(0.5, 1.0, 0.01):
            estimator = AffinityPropagation(
                    affinity='precomputed',
                    preference=preference,
                    damping=damping)
            estimator.fit(similarity)
            score = adjusted_rand_score(mentions, estimator.labels_)
            print('score=', score, 'preference=', preference, 'damping=', damping)
            if score > best_score:
                best_score = score
                best_params[0] = preference
                best_params[1] = damping

    def fun(x, gold=None, similarity=None):
        preference = x[0]
        damping = x[1]
        estimator = AffinityPropagation(
                affinity='precomputed',
                preference=preference,
                damping=damping)
        estimator.fit(similarity)
        score = adjusted_rand_score(gold, estimator.labels_)
        print('score=', score, 'preference=', preference, 'damping=', damping)
        return -1. * score

    res = minimize(fun,
                   best_params,
                   bounds=[(-100, -0.1), (0.501, 0.998)],
                   args=(mentions, similarity),
                   tol=1e-8
                   )
    print('Result:', res)

    estimator = AffinityPropagation(
            affinity='precomputed', preference=res.x[0], damping=res.x[1])
    estimator.fit(similarity)
    print('Rand score:', adjusted_rand_score(mentions, estimator.labels_))

    # 6. write back cluster id to the mentions
    mention_id = 0
    for sentence in dataset:
        for token in sentence:
            if token.COREF != '_':
                token.COREF = '{}'.format(estimator.labels_[mention_id])
                mention_id += 1

    # import hdbscan
    # clusterer = hdbscan.HDBSCAN(
    #         metric='precomputed',
    #         min_cluster_size=2,
    #         gen_min_span_tree=True
    #         )
    # clusterer.fit(similarity)
    # clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    # plt.show()
    # clusterer.condensed_tree_.plot()
    # plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    # 1. load conll file
    input_set = ConlluDataset(args.conll)

    # 2. build key file to score against
    build_scoring_file(input_set, args.conll, 'clustering.key')

    # 3. pre-process the dependency tree to unfold coordination
    #   and convert the gold span based mentions to head-based mentions
    for sentence in input_set:
        preprocess_sentence(sentence)

    # 4. this is where we should do our coref resolution
    cluster_mentions(input_set)

    # 5. post-process the sentence to get spans again
    for sentence in input_set:
        postprocess_sentence(sentence)

    # 6. write out our sentence for scoring
    build_scoring_file(input_set, args.conll, 'clustering.response')



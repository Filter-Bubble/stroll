import argparse
import logging

import numpy as np
import torch
import dgl

from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_rand_score

from scipy.optimize import minimize

from stroll.model import Net
from stroll.graph import ConlluDataset, GraphDataset
from stroll.coref import preprocess_sentence, postprocess_sentence
from stroll.labels import FasttextEncoder

from stroll.coref import mentions_match_exactly
from stroll.coref import mentions_match_relaxed

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
                if token.COREF != '_':
                    logging.error(
                            'Hidden token has a coref={}'.format(token.COREF)
                            )
                    print(sentence)
                    print()
                continue
            if token.COREF != '_':
                coref = token.COREF
            else:
                coref = '-'
            keyfile.write('{}\t0\t{}\t{}\t{}\n'.format(
                sentence.doc_id, token.ID, token.FORM, coref))

    keyfile.write('#end document\n')
    keyfile.close()


def similarity_from_net(dataset):
    name = 'runs/ADAMmm_minnorm_1e-02_25b_FL1.5_128d_2lUPOS_FEATS_DEPREL_WVEC_FT100/model_000187276.pt'
    logging.info('Loading model "{}"'.format(name))
    state_dict = torch.load(name)
    params = state_dict['hyperparams'].__dict__

    net = Net(
        in_feats=state_dict['embedding.fc.0.weight'].shape[1],
        h_layers=params['h_layers'],  # args.h_layers,
        h_dims=params['h_dims'],
        out_feats_a=2,  # role / no-role
        out_feats_b=128,  # mention vector
        activation='relu'
        )
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    logging.info('Building graph')
    graph_set = GraphDataset(
            dataset=dataset,
            sentence_encoder=FasttextEncoder(params['fasttext']),
            features=params['features']
            )
    test_graph = dgl.batch([g for g in graph_set])

    logging.info('Running net')
    id_out, match_out = net(test_graph)
    _, system = torch.max(id_out, dim=1)

    # get all gold coref annotations
    coref = test_graph.ndata['coref'].view(-1)

    # pick the actual mentions
    correct_idx = np.where(coref >= 0)

    # get the correct clusters
    correct = coref[correct_idx]
    nmentions = len(correct)

    # get the pairwise distances for the system's mentions,
    # assuming gold mentions
    match_out = match_out[correct_idx]
    pdist = torch.pdist(match_out)
    pdist = 50. * pdist.pow(2) / pdist.pow(2).median()

    sent_dist = test_graph.ndata['index'].view(-1).unsqueeze(1).float()
    sent_dist = sent_dist[correct_idx]
    sent_dist = torch.pdist(sent_dist, p=1.0)

    # build affinity matrix
    affinities = np.zeros([nmentions, nmentions])

    for m1 in range(nmentions):
        for m2 in range(m1):
            m1m2 = nmentions*m2 - m2*(m2+1)/2 + m1 - 1 - m2
            m1m2 = int(m1m2)
            affinities[m1, m2] = - pdist[m1m2] - 1.0 * sent_dist[m1m2]
            affinities[m2, m1] = - pdist[m1m2] - 1.0 * sent_dist[m1m2]

    affinities -= 5.0
    for m1 in range(nmentions):
        affinities[m1, m1] += 4.0

    return affinities.clip(-100.0, -1.0)


def similarity_from_gold(mentions):
    similarity = np.zeros([len(mentions), len(mentions)])
    for m1, entity in enumerate(mentions):
        similarity[m1, m1] = -5.0
        for m2 in range(m1):
            if abs(m1-m2) < 1025 and mentions[m2] == mentions[m1]:
                similarity[m1, m2] = -1.
                similarity[m2, m1] = -1.
            else:
                similarity[m1, m2] = -100.
                similarity[m2, m1] = -100.

    return similarity


def gridsearch_and_optimize(similarity, mentions):
    best_params = np.zeros([2])
    best_score = 0
    for preference in np.arange(-100, -1, 5):
        for damping in np.arange(0.5, 1.0, 0.01):
            estimator = AffinityPropagation(
                    affinity='precomputed',
                    preference=preference,
                    damping=damping)
            estimator.fit(similarity)
            print('Similarity matrix:', np.shape(similarity))
            print(similarity)
            print('Mentions:', len(mentions), mentions)
            score = adjusted_rand_score(mentions, estimator.labels_)
            print(
                    'score=', score,
                    'preference=', preference,
                    'damping=', damping
                    )
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


def cluster_mentions(dataset):
    mentions = []
    # 1. find all mentions
    for sentence in dataset:
        for token in sentence:
            print('Token:', token.FORM, token.COREF)
            if token.COREF != '_':
                mentions.append(token.COREF)
    print('Found {} mentions.'.format(len(mentions)))

    # 2a. make similarity matrix
    similarity_g = similarity_from_gold(mentions)
    print('Gold similarities avg, min, max:',
          similarity_g.mean(),
          similarity_g.min(),
          similarity_g.max()
          )

    similarity_s = similarity_from_net(dataset)
    print('System similarities avg, min, max:',
          similarity_s.mean(),
          similarity_s.min(),
          similarity_s.max()
          )

    # 2b. add noise?
    # similarity += np.random.normal(scale=5, size=len(mentions)**2).reshape(
    #         [len(mentions), len(mentions)]
    #         )

    # 3. plot it
    # plt.imshow(similarity, cmap='hot', interpolation='nearest')
    # plt.show()

    # 5. try clustering approaches
    estimator = AffinityPropagation(
            affinity='precomputed', preference=-80, damping=0.7)
    estimator.fit(similarity_g)
    print('G Rand score:', adjusted_rand_score(mentions, estimator.labels_))
    estimator.fit(similarity_s)
    print('S Rand score:', adjusted_rand_score(mentions, estimator.labels_))

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


def external_scoring(args):
    # 1. load conll file
    input_set = ConlluDataset(args.conll)

    # 2. build key file to score against
    build_scoring_file(input_set, args.conll, 'clustering.key')

    # 3. pre-process the dependency tree to unfold coordination
    #   and convert the gold span based mentions to head-based mentions
    for sentence in input_set:
        preprocess_sentence(sentence)

    # 4. this is where we should do our coref resolution
    # cluster_mentions(input_set)

    # 5. post-process the sentence to get spans again
    for sentence in input_set:
        postprocess_sentence(sentence)

    # 6. write out our sentence for scoring
    build_scoring_file(input_set, args.conll, 'clustering.response')


def internal_scoring(args):
    # 1. load conll file
    input_set = ConlluDataset(args.conll)

    logging.info('Processing')
    targets = []
    candidates = []
    for sentence in input_set:
        # 2. pre-process the dependency tree to unfold coordination
        #    keep track of both target and candidates
        m_braket, m_head = preprocess_sentence(sentence)
        targets += m_braket
        candidates += m_head

        # 3. run network
        # 4. get the system mentions

    # 5. score
    logging.info('Scoring')
    print('Gold:         ', len(targets))
    print('System:       ', len(candidates))

    # First find exact matches
    exact_match = 0
    logging.info('Scoring exact')

    unmatched_targets = []
    while len(targets):
        target_matched = False
        target = targets.pop()
        for candidate in candidates:
            if target.sentence != candidate.sentence:
                continue
            if mentions_match_exactly(target, candidate):
                exact_match += 1
                target_matched = True
                candidates.remove(candidate)
                break
        if not target_matched:
            unmatched_targets.append(target)

    # Then find relaxed matches
    relaxed_match = 0
    logging.info('Scoring relaxed')

    targets = unmatched_targets
    unmatched_targets = []
    while len(targets):
        target_matched = False
        target = targets.pop()
        for candidate in candidates:
            if target.sentence != candidate.sentence:
                continue

            if mentions_match_relaxed(target, candidate):
                relaxed_match += 1
                target_matched = True
                candidates.remove(candidate)
                break
        if not target_matched:
            unmatched_targets.append(target)

    # Report
    print('Exact match:  ', exact_match)
    print('Relaxed match:', relaxed_match)
    print('Missed:       ', len(unmatched_targets))
    print('Invented:     ', len(candidates))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    internal_scoring(args)

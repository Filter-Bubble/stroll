import argparse
import logging

import torch

from jinja2 import FileSystemLoader, Environment

from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support as PRF

from scorch.scores import muc, b_cubed, ceaf_e

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence
from stroll.coref import mark_gold_anaphores
from stroll.coref import nearest_linking
from stroll.coref import predict_anaphores, predict_similarities
from stroll.coref import coref_collate
from stroll.graph import GraphDataset
from stroll.model import CorefNet
from stroll.labels import FasttextEncoder
from stroll.evaluate import clusters_to_sets

MAX_MENTION_DISTANCE = 20
MAX_MENTION_PER_DOC = 1000


torch.manual_seed(43)

parser = argparse.ArgumentParser(
        description='Run a mention linking, R-GCN for Coreference resolution.'
        )
parser.add_argument(
        '--model',
        dest='model_name',
        default='models/coref.pt',
        help='Trained CorefNet to use',
        )
parser.add_argument(
        '--output',
        help='Output file in conllu format',
        )
parser.add_argument(
        '--html',
        help='Output file in html format',
        )
parser.add_argument(
        '--score',
        action='store_true',
        default=False,
        help='Score coreference using annotation from the input'
        )
parser.add_argument(
        'input',
        help='input CoNLL file',
        )


def write_html(dataset, name):
    loader = FileSystemLoader('.')
    env = Environment(loader=loader)
    template = env.get_template('highlighted.template')

    documents = {}
    entities = {}
    for sentence in dataset:
        doc_id = sentence.doc_id
        if doc_id in documents:
            documents[doc_id]['sentences'].append(sentence)
        else:
            documents[doc_id] = {
                    'doc_id': doc_id,
                    'sentences': [sentence]
                    }
        for token in sentence:
            if token.COREF != '_':
                entities[token.COREF] = 1

    with open(name, 'w') as f:
        f.write(template.render(
            documents=list(documents.values()),
            entities=list(entities.keys())
            )
        )


def main(args):
    logging.basicConfig(level=logging.INFO)

    # 1. load the CorefNet configuration
    state_dict = torch.load(args.model_name)
    hyperparams = state_dict.pop('hyperparams')
    if 'WVEC' in hyperparams.features:
        if hyperparams.fasttext:
            sentence_encoder = FasttextEncoder(hyperparams.fasttext)
    else:
        sentence_encoder = None

    # 2. load conll file
    dataset = ConlluDataset(args.input)

    # 3. pre-process the dependency tree to unfold coordination
    #   and convert the gold span based mentions to head-based mentions
    # for sentence in dataset:
    #     preprocess_sentence(sentence)

    # when evaluating
    mark_gold_anaphores(dataset)

    # 4. make graphs from the conll dataset
    graphset = GraphDataset(
            dataset=dataset,
            sentence_encoder=sentence_encoder,
            features=hyperparams.features
            )
    graph_loader = DataLoader(
        graphset,
        num_workers=2,
        batch_size=len(graphset),
        collate_fn=coref_collate
        )

    # 5. initialize the network
    net = CorefNet(
        in_feats=graphset.in_feats,
        in_feats_a=75 + 5,  # mention + mention_pair
        in_feats_b=(hyperparams.h_dims + 75) * 2 + 5,
        h_layers=hyperparams.h_layers,
        h_dims=hyperparams.h_dims,
        activation='relu'
        )
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    torch.no_grad()

    # 6. run coreference
    for test_graph, test_mentions in graph_loader:
        gvec = net(test_graph)

        test_clusters = [mention.refid for mention in test_mentions]

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
                similarities,
                anaphores,  # [mention.anaphore for mention in test_mentions]
                )

        # save mentions in the dataset
        # clear old info
        for s in dataset:
            for t in s:
                t.COREF = '_'

        sent_index = test_graph.ndata['sent_index'][mention_idxs]
        token_index = test_graph.ndata['token_index'][mention_idxs]
        for s, t, m in zip(sent_index, token_index, system_clusters):
            dataset[s][t].COREF = '{:d}'.format(int(m))

        if args.score:
            # score the clustering
            system_set = clusters_to_sets(system_clusters)
            gold_set = clusters_to_sets(test_clusters)

            muc_rpf = muc(gold_set, system_set)
            b3_rpf = b_cubed(gold_set, system_set)
            ce_rpf = ceaf_e(gold_set, system_set)

            # score the anaphores
            targets = [mention.anaphore for mention in test_mentions]

            ana_scores = PRF(
                    targets,
                    torch.round(anaphores).detach().numpy(),
                    labels=[1.0]
                    )
            conll = (muc_rpf[2] + b3_rpf[2] + ce_rpf[2]) / 3.0
            print('muc', muc_rpf)
            print('b3', b3_rpf)
            print('ceafe', ce_rpf)
            print('conll', conll)
            print('anaphores:', ana_scores)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(dataset.__repr__())
    if args.html:
        write_html(dataset, args.html)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

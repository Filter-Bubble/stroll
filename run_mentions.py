import argparse
import logging

import torch
import dgl

from sklearn.metrics import precision_recall_fscore_support as PRF

from torch.utils.data import DataLoader

from stroll.conllu import ConlluDataset
from stroll.graph import GraphDataset
from stroll.coref import preprocess_sentence, postprocess_sentence
from stroll.coref import get_mentions

from stroll.model import MentionNet
from stroll.labels import FasttextEncoder

# TODO: always label arguments ArgN as a mention?

parser = argparse.ArgumentParser(
        description='Run a mention detection network'
        )
parser.add_argument(
        'input',
        help='Input file in CoNLL format'
        )
parser.add_argument(
        '--model',
        default='models/mentions.pt',
        help='Trained MentionNet to use',
        )
parser.add_argument(
        '--conll2012',
        help='Output file in conll2012 format',
        )
parser.add_argument(
        '--score',
        help='Score using gold annotation from the input',
        default=False,
        action='store_true'
        )
parser.add_argument(
        '--verbose',
        help='Print gold and found mentions',
        default=False,
        action='store_true'
        )
parser.add_argument(
        '--output',
        help='Output file in conllu format',
        )


def write_output_conll2012(dataset, filename):
    keyfile = open(filename, 'w')

    firstDoc = True
    current_doc = None
    for sentence in dataset:
        if sentence.doc_id != current_doc:
            if firstDoc:
                firstDoc = False
            else:
                keyfile.write('#end document\n')

            current_doc = sentence.doc_id
            keyfile.write('#begin document ({});\n'.format(current_doc))
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    # 1. load the MentionNet configuration
    state_dict = torch.load(args.model)
    hyperparams = state_dict.pop('hyperparams')
    if 'WVEC' in hyperparams.features:
        if hyperparams.fasttext:
            sentence_encoder = FasttextEncoder(hyperparams.fasttext)
    else:
        sentence_encoder = None

    # 2. load conll file
    dataset = ConlluDataset(args.input)

    # 3. pre-process the dependency tree to unfold coordination
    #    and convert the gold span based mentions to head-based mentions
    gold_braket = []
    gold_head = []

    refid_lookup = []  # by sent_index
    for sentence in dataset:
        braket, head = preprocess_sentence(sentence)
        gold_braket += braket
        gold_head += head

        head_by_token_index = {}
        for mention in gold_head:
            sentence = mention.sentence
            index = sentence.index(mention.head)
            head_by_token_index[index] = mention.refid
        refid_lookup.append(head_by_token_index)

    if args.verbose and args.score:
        print('Number of mentions (braket): {}'.format(len(gold_braket)))
        print('Number of mentions (head):   {}'.format(len(gold_head)))
        for mention in gold_head:
            print(mention)
            print('----')

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
        collate_fn=dgl.batch
        )

    # 5. initialize the network
    net = MentionNet(
        in_feats=graphset.in_feats,
        h_layers=hyperparams.h_layers,
        h_dims=hyperparams.h_dims,
        activation='relu'
        )
    net.load_state_dict(state_dict, strict=False)

    net.eval()

    # 6. score mentions
    entity = 1000
    for g in graph_loader:
        xa = net(g)

        # system mentions
        _, system = torch.max(xa, dim=1)

        # save mentions in the dataset
        sent_indices = g.ndata['sent_index'].tolist()
        token_indices = g.ndata['token_index'].tolist()
        for sent_index, token_index, isMention in \
                zip(sent_indices, token_indices, system):
            if isMention:
                if token_index in refid_lookup[sent_index]:
                    dataset[sent_index][token_index].COREF_HEAD = \
                            refid_lookup[sent_index][token_index]
                else:
                    # treat every mention as a new entity
                    dataset[sent_index][token_index].COREF_HEAD = \
                            '{}'.format(entity)
                    entity += 1
            else:
                dataset[sent_index][token_index].COREF_HEAD = '_'

        if args.score:
            # correct mentions:
            target = g.ndata['coref'].view(-1).clamp(0, 1)

            # score
            score_id_p, score_id_r, score_id_f1, support = PRF(
                target, system, labels=[1]
                )

            print('P: {}'.format(score_id_p[0]))
            print('R: {}'.format(score_id_r[0]))
            print('F: {}'.format(score_id_f1[0]))
            print('support: {}'.format(support))
            print('Failed look up refid for {} mentions'.format(entity - 1000))

    if args.verbose:
        print('Found mentions:')
        for sentence in dataset:
            mentions = get_mentions(sentence)
            for mention in mentions:
                print(mention)
                print('----')

    # 3. Output: head based to conll
    if args.output:
        with open(args.output, 'w') as f:
            f.write(dataset.__repr__())

    # 4. Output: span based conll2012
    if args.conll2012:
        for sentence in dataset:
            postprocess_sentence(sentence)
        write_output_conll2012(dataset, args.output)

import torch  # script version
import argparse

import dgl
from torch.utils.data import DataLoader

from stroll.graph import GraphDataset
from stroll.model import Net
from stroll.labels import BertEncoder, FasttextEncoder
from stroll.evaluate import evaluate


parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument(
        '--batch_size',
        dest='batch_size',
        default=50,
        help='Evaluation batch size.'
        )
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


if __name__ == '__main__':
    args = parser.parse_args()

    state_dict = torch.load(args.model_name)
    hyperparams = state_dict.pop('hyperparams')

    if 'WVEC' in hyperparams.features:
        if hyperparams.fasttext:
            sentence_encoder = FasttextEncoder(hyperparams.fasttext)
        else:
            sentence_encoder = BertEncoder()
    else:
        sentence_encoder = None

    eval_set = GraphDataset(
            args.dataset,
            sentence_encoder=sentence_encoder,
            features=hyperparams.features
            )
    evalloader = DataLoader(
            eval_set,
            batch_size=args.batch_size,
            collate_fn=dgl.batch
            )

    net = Net(
            in_feats=eval_set.in_feats,
            h_layers=hyperparams.h_layers,
            h_dims=hyperparams.h_dims,
            out_feats_a=2,
            out_feats_b=21,
            activation='relu'
            )
    net.load_state_dict(state_dict)

    fig_name = args.model_name[:-2]
    evaluate(net, evalloader, fig_name, batch_size=50)

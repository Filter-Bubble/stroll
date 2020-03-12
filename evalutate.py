import argparse

import dgl
import torch
from torch.utils.data import DataLoader

from stroll.graph import GraphDataset
from stroll.model import Net
from stroll.labels import BertEncoder, frame_codec, role_codec

from sklearn.metrics import confusion_matrix, classification_report

from progress.bar import Bar

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
        '--cm_normalize',
        choices=['true', 'pred', 'all', 'none'],
        default='none',
        help='Normalization for the confusion matrix.'
        )
parser.add_argument(
        'dataset',
        help='Evaluation dataset in conllu format',
        )
args = parser.parse_args()

if args.cm_normalize == 'none':
    args.cm_normalize = None

od = torch.load(args.model_name)
hyperparams = od.pop('hyperparams')


if 'WVEC' in hyperparams.features:
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
net.load_state_dict(od)

predicted_frames = []
gold_frames = []

predicted_roles = []
gold_roles = []

progbar = Bar('Evaluating', max=len(eval_set))

net.eval()
with torch.no_grad():
    for g in evalloader:
        lf, lr = net(g)
        _, pf = torch.max(lf, dim=1)
        _, pr = torch.max(lr, dim=1)

        gf = g.ndata['frame']
        gr = g.ndata['role']

        predicted_frames += frame_codec.inverse_transform(pf).tolist()
        gold_frames += frame_codec.inverse_transform(gf).tolist()

        predicted_roles += role_codec.inverse_transform(pr).tolist()
        gold_roles += role_codec.inverse_transform(gr).tolist()

        progbar.next(args.batch_size)

progbar.finish()

print('Frames')
print(confusion_matrix(
    gold_frames, predicted_frames, normalize=args.cm_normalize
    ))

print('\n')

print(classification_report(gold_frames, predicted_frames))

print('\n -- \n')

print('Roles')
print(confusion_matrix(
    gold_roles, predicted_roles, normalize=args.cm_normalize
    ))

print('\n')

print(classification_report(gold_roles, predicted_roles))

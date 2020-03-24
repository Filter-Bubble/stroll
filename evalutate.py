import argparse

import dgl
import torch
from torch.utils.data import DataLoader

from stroll.graph import GraphDataset
from stroll.model import Net
from stroll.labels import BertEncoder, FasttextEncoder
from stroll.labels import frame_codec, role_codec

from sklearn.metrics import confusion_matrix, classification_report

from progress.bar import Bar
import seaborn as sns
import matplotlib.pyplot as plt

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
args = parser.parse_args()

od = torch.load(args.model_name)
hyperparams = od.pop('hyperparams')

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
net.load_state_dict(od)

predicted_frames = []
gold_frames = []

predicted_roles1 = []
predicted_roles2 = []
predicted_roles3 = []
gold_roles = []

progbar = Bar('Evaluating', max=len(eval_set))

net.eval()
with torch.no_grad():
    for g in evalloader:
        lf, lr = net(g)
        lowest = torch.min(lr)

        # Get best frame
        _, pf = torch.max(lf, dim=1)

        # Get best role, and set its score to zero
        _, pr1st = torch.max(lr, dim=1)
        for i in range(len(g)):
            lr[i, pr1st[i]] = lowest.item()

        # Get second best role, and set its score to zero
        _, pr2nd = torch.max(lr, dim=1)
        for i in range(len(g)):
            lr[i, pr2nd[i]] = lowest.item()

        # Get third best role
        _, pr3rd = torch.max(lr, dim=1)

        gf = g.ndata['frame']
        gr = g.ndata['role']

        predicted_frames += frame_codec.inverse_transform(pf).tolist()
        gold_frames += frame_codec.inverse_transform(gf).tolist()

        predicted_roles1 += role_codec.inverse_transform(pr1st).tolist()
        predicted_roles2 += role_codec.inverse_transform(pr2nd).tolist()
        predicted_roles3 += role_codec.inverse_transform(pr3rd).tolist()
        gold_roles += role_codec.inverse_transform(gr).tolist()

        progbar.next(args.batch_size)

progbar.finish()

main_args = ['Arg0', 'Arg1', 'Arg2', 'Arg3', 'Arg4', 'Arg5']
reduced_gold = []
for label in gold_roles:
    if label in main_args:
        reduced_gold.append('Arg')
    elif label == '_':
        reduced_gold.append('_')
    else:
        reduced_gold.append('Mod')

reduced_pred = []
for label in predicted_roles1:
    if label in main_args:
        reduced_pred.append('Arg')
    elif label == '_':
        reduced_pred.append('_')
    else:
        reduced_pred.append('Mod')

norm = None
labels = role_codec.classes_
conf_frames = confusion_matrix(gold_frames, predicted_frames, normalize=norm)
conf_roles1 = confusion_matrix(gold_roles, predicted_roles1, labels=labels, normalize=norm)
conf_roles2 = confusion_matrix(gold_roles, predicted_roles2, labels=labels, normalize=norm)
conf_roles3 = confusion_matrix(gold_roles, predicted_roles3, labels=labels, normalize=norm)
conf_reduced = confusion_matrix(reduced_gold, reduced_pred, normalize=norm)

#

print(classification_report(gold_frames, predicted_frames))
print(classification_report(gold_roles, predicted_roles1))
print(classification_report(reduced_gold, reduced_pred))

print('\n -- \n')

print('Frames')
print(conf_frames)
print('Roles - best')
print(conf_roles1)
print('\n')
print('Roles - second')
print(conf_roles2)
print('\n')
print('Roles - third')
print(conf_roles3)
print('\n')
print('Roles - simplified')
print(conf_reduced)

# Calculate the normalized confusion matrix
norm = 'true'
conf_frames = confusion_matrix(gold_frames, predicted_frames, normalize=norm)
conf_roles1 = confusion_matrix(gold_roles, predicted_roles1, labels=labels, normalize=norm)
conf_roles2 = confusion_matrix(gold_roles, predicted_roles2, labels=labels, normalize=norm)
conf_roles3 = confusion_matrix(gold_roles, predicted_roles3, labels=labels, normalize=norm)
conf_reduced = confusion_matrix(reduced_gold, reduced_pred, normalize=norm)

# take the model fileanme (without pt) as figure name
fmt = "3.0f"
fig_name = args.model_name[0:-2]

figure = plt.figure(figsize=[10., 10.])
sns.heatmap(
        100. * conf_roles1, fmt=fmt, annot=True, cbar=False,
        cmap="Greens", xticklabels=labels, yticklabels=labels
        )
plt.savefig(fig_name + 'roles1.png')

figure = plt.figure(figsize=[10., 10.])
sns.heatmap(
        100. * conf_roles2, fmt=fmt, annot=True, cbar=False,
        cmap="Greens", xticklabels=labels, yticklabels=labels
        )
plt.savefig(fig_name + 'roles2.png')

figure = plt.figure(figsize=[10., 10.])
sns.heatmap(
        100. * conf_roles3, fmt=fmt, annot=True, cbar=False,
        cmap="Greens", xticklabels=labels, yticklabels=labels
        )
plt.savefig(fig_name + 'roles3.png')

figure = plt.figure(figsize=[10., 10.])
sns.heatmap(
        100. * conf_reduced, fmt=fmt, annot=True, cbar=False, cmap="Greens",
        xticklabels=['Arg', 'ArgM', '_'], yticklabels=['Arg', 'Mod', '_']
        )
plt.savefig(fig_name + 'roles_red.png')

figure = plt.figure(figsize=[10., 10.])
labels = frame_codec.classes_
sns.heatmap(
        100. * conf_frames, fmt=fmt, annot=True, cbar=False, cmap="Greens",
        xticklabels=labels, yticklabels=labels
        )
plt.savefig(fig_name + 'frames.png')

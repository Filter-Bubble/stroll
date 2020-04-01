import argparse
import logging

import torch
from torch.utils.data import DataLoader

import dgl

import numpy as np

from stroll.model import Net
from stroll.graph import GraphDataset
from stroll.evaluate import evaluate
from stroll.labels import FasttextEncoder, \
        ROLE_TARGET_DISTRIBUTIONS, FRAME_TARGET_DISTRIBUTIONS

import scipy as sp

parser = argparse.ArgumentParser(
        description='Construct an ensemble.'
        )
parser.add_argument(
        '--member',
        nargs='*',
        dest='members',
        default=[
            'runs/best/ba_022980256.pt',
            'runs/best/ce_026175619.pt',
            'runs/best/focal_015025552.pt',
            'runs/best/h2_007565756.pt',
            'runs/best/kl_013730138.pt'
            ],
        help='Members in the ensemble'
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
        default='test.conllu',
        help='Test dataset in conllu format',
        )
parser.add_argument(
        '--name',
        default='myensemble',
        help='Name of the ensemble'
        )
parser.add_argument(
        '--estimate_kl',
        dest='estimate_kl',
        default=False,
        action='store_true',
        help='Estimate the KL divergence per ensemble member'
        )


class Ensemble():
    def __init__(self, name='ensemble'):
        self.members = []
        self.role_weights = []
        self.frame_weights = []
        self.state_dicts = []
        self.name = name

    def add_member(self, net, state_dict,
                   role_weight=1.0, frame_weight=1.0, name='anon'):
        self.members.append(net)
        self.role_weights.append(role_weight)
        self.frame_weights.append(frame_weight)
        self.state_dicts.append(state_dict)
        self.name = name

    def set_weights(self, role_weights=None, frame_weights=None):
        if role_weights is not None:
            if (len(role_weights) != len(self.members)):
                print('Incorrect number of role weights')
            else:
                self.role_weights = role_weights

        if frame_weights is not None:
            if (len(frame_weights) != len(self.members)):
                print('Incorrect number of frame weights')
            else:
                self.frame_weights = frame_weights

    def load_member(self, name):
        logging.info('Loading model "{}"'.format(name))
        state_dict = torch.load(member)
        params = state_dict['hyperparams'].__dict__

        net = Net(
            in_feats=state_dict['embedding.fc.0.weight'].shape[1],
            h_layers=params['h_layers'],  # args.h_layers,
            h_dims=params['h_dims'],
            out_feats_a=2,  # number of frames
            out_feats_b=21,  # number of roles
            activation=params['activation']
            )

        net.load_state_dict(state_dict, strict=False)
        net.eval()

        ensemble.add_member(net, params, name=name)

    def eval(self):
        for member in self.members:
            member.eval()

    def __call__(self, g):
        logp_frame = torch.zeros([len(g), 2])
        logp_role = torch.zeros([len(g), 21])

        for frame_weight, role_weight, member in zip(
                self.frame_weights, self.role_weights, self.members):
            # shapes [N, C]
            logits_F, logits_R = member(g)

            logpred_F = torch.log_softmax(logits_F, dim=1).detach()
            logpred_R = torch.log_softmax(logits_R, dim=1).detach()

            logp_frame += (logpred_F * frame_weight)
            logp_role += (logpred_R * role_weight)

        # TODO: normalize
        return torch.exp(logp_frame), torch.exp(logp_role)

    def __len__(self):
        return len(self.members)


ensemble = Ensemble()


def KL(logq, logp):
    """Shape of truth q and estimate p are expected to be [N, C]"""
    return torch.sum(
            - torch.exp(logq) * logp
            + torch.exp(logq) * logq
            ).item()


def ensemble_klab(wa, KLab):
    """
    KLab = KL(a, b) + K(b, a) for a =/= b
         = KL(q, a)               a === b
    """
    result = 0.0
    for i in range(len(ensemble)):
        for j in range(len(ensemble)):
            if i != j:
                result -= 0.25 * (wa[i] * wa[j] * KLab[i, j])
            else:
                result += KLab[i, i] * wa[i]

    return result  # * 1e-6


def minimize_ensemble_kl(klf, klr):
    constraint_matrix = np.ones([len(ensemble), len(ensemble)])
    wa_sum_one = sp.optimize.LinearConstraint(constraint_matrix, 1.0, 1.0)
    atol = 1e-4
    seed = 43
    maxiter = 100000

    fmin = sp.optimize.differential_evolution(
            lambda x: ensemble_klab(x, klf),
            [(0., 1.0)]*len(ensemble),
            constraints=wa_sum_one,
            disp=False,
            polish=True,
            seed=seed,
            maxiter=maxiter,
            atol=atol
            )
    rmin = sp.optimize.differential_evolution(
            lambda x: ensemble_klab(x, klr),
            [(0., 1.0)]*len(ensemble),
            constraints=wa_sum_one,
            disp=False,
            polish=True,
            seed=seed,
            maxiter=maxiter,
            atol=atol
            )

    return fmin, rmin


def estimate_ensemble_klab(trainloader):
    klr = torch.zeros([len(ensemble), len(ensemble)])
    klf = torch.zeros([len(ensemble), len(ensemble)])

    logq_all_roles = torch.log_softmax(
            ROLE_TARGET_DISTRIBUTIONS,
            dim=1).detach()
    logq_all_frames = torch.log_softmax(
            FRAME_TARGET_DISTRIBUTIONS,
            dim=1).detach()

    word_count = 0
    for g in trainloader:
        logp_role = []
        logp_frame = []
        word_count += len(g)

        for member in ensemble:
            # shapes [N, C]
            logits_F, logits_R = member(g)

            logpred_F = torch.log_softmax(logits_F, dim=1).detach()
            logpred_R = torch.log_softmax(logits_R, dim=1).detach()

            logp_frame.append(logpred_F)
            logp_role.append(logpred_R)

        # shape [N, C]
        logq_role = logq_all_roles[g.ndata['role'].view(-1).long()]
        logq_frame = logq_all_frames[g.ndata['frame'].view(-1).long()]

        for i in np.arange(len(ensemble)):
            for j in np.arange(len(ensemble)):
                if i == j:
                    klr[i, i] += KL(logq_role, logp_role[i])
                    klf[i, i] += KL(logq_frame, logp_frame[i])
                else:
                    klr[i, j] += KL(logp_role[i], logp_role[j])
                    klr[i, j] += KL(logp_role[j], logp_role[i])

                    klf[i, j] += KL(logp_frame[i], logp_frame[j])
                    klf[i, j] += KL(logp_frame[j], logp_frame[i])

        print("Frames:\n", klf)
        print("Roles:\n", klr)
        print('- {:08d} -'.format(word_count))

        return klf, klr


# runs/best/ba_022980256.pt
# runs/best/ce_026175619.pt
# runs/best/focal_015025552.pt
# runs/best/h2_007565756.pt
# runs/best/kl_013730138.pt

klf5 = torch.Tensor(
        [[344742.9375,   94464.1016,   21583.6445,   30332.9609,   13562.8701],
         [94464.1016, 2369102.5000,   36378.5469,  196317.5156,  153391.1719],
         [21583.6445,   36378.5430,  689545.4375,   55250.1133,   37331.3203],
         [30332.9551,  196317.5000,   55250.1133,  413628.5312,   27750.2188],
         [13562.8701,  153391.1562,   37331.3242,   27750.2188,  250818.2031]]
        )
klr5 = torch.Tensor(
        [[7.3058e+02, 7.3226e+06, 5.3819e+06, 7.9990e+05, 7.4167e+02],
         [7.3226e+06, 6.4295e+06, 3.0334e+05, 4.8711e+06, 7.3339e+06],
         [5.3819e+06, 3.0334e+05, 4.5230e+06, 3.3701e+06, 5.3911e+06],
         [7.9990e+05, 4.8711e+06, 3.3701e+06, 5.6972e+05, 8.0005e+05],
         [7.4167e+02, 7.3339e+06, 5.3911e+06, 8.0005e+05, 1.0727e+03]]
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    for member in args.members:
        ensemble.load_member(member)

    # TODO: don't assume a uniform ensemble
    params = ensemble.state_dicts[0]
    encoder = FasttextEncoder(params['fasttext'])
    features = params['features']
    batch_size = params['batch_size']

    train_set = GraphDataset(
            args.train_set,
            sentence_encoder=encoder,
            features=features,
            )
    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dgl.batch
        )

    test_set = GraphDataset(
            args.test_set,
            sentence_encoder=encoder,
            features=features,
            )
    testloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dgl.batch
        )

    if(args.estimate_kl):
        klf, klr = estimate_ensemble_klab(trainloader)
    else:
        logging.warn(
          'Using precalucated weights; only works for the default ensemble'
                    )
        klf = klf5
        klr = klr5

    fmin, rmin = minimize_ensemble_kl(klf.numpy(), klr.numpy())
    print(fmin, rmin)
    ensemble.set_weights([1., 0., 0., 1., 1.], [1., 0., 0., 1., 1.])
    ensemble.name = args.name
    evaluate(ensemble, testloader, args.name, batch_size=50)

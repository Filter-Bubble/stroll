# Taken from https://github.com/clcarwin/focal_loss_pytorch.github
# License: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self, weights=None):
        super(CrossEntropy, self).__init__()
        self.weights = weights

    def forward(self, input, target):
        return F.cross_entropy(
                input.view(1, 2, -1), target.view(1, -1), self.weights
                )


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = None
        self.size_average = size_average

    def forward(self, input, target):
        """Input shape [C,N], target shape [N]"""

        target = target.long()

        logpt = F.log_softmax(input, dim=0)
        logpt = logpt.gather(0, target.view(1, -1))
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class HingeSquared(nn.Module):
    def __init__(self):
        super(HingeSquared, self).__init__()
        self.yhat = 2 * torch.eye(21) - torch.ones([21, 21])

    def forward(self, input, target):
        """Input shape [C,N], target shape [N]"""

        yhat = self.yhat[target.long()].transpose(0, 1)
        HL_CN = 0.5 - yhat * input
        HL_CN[HL_CN < 0.] = 0.
        HL_CN = HL_CN**2

        return torch.sum(HL_CN)


# https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
class KullbackLeibler(nn.Module):
    def __init__(self, target_distribution=None):
        super(KullbackLeibler, self).__init__()
        # The desired distribution peaks at the right label
        d = torch.eye(21)

        # Labels 0 - 4 are Arg[0-5], and are similar
        d[0:6, 0:6] += 0.01

        # Labels 5 - 19 are ArgM, and are similar
        d[6:20, 6:20] += 0.01

        self.logq = F.log_softmax(target_distribution, dim=0)

    def forward(self, input, target):
        """Input shape [C,N], target shape [N]"""

        logq = self.logq[target.long()].transpose(0, 1)
        logp = F.log_softmax(input, dim=0)
        p = torch.exp(logp)

        KL_N = torch.sum(p * logp - p * logq, dim=0)
        return torch.sum(KL_N)


# https://en.wikipedia.org/wiki/Bhattacharyya_distance
class Bhattacharyya(nn.Module):
    def __init__(self, target_distribution=None):
        super(Bhattacharyya, self).__init__()
        self.q = F.softmax(target_distribution, dim=0)

    def forward(self, input, target):
        """Input shape [C,N], target shape [N]"""

        q = self.q[target.long()].transpose(0, 1)
        p = F.softmax(input, dim=0)
        spq_N = torch.sum(torch.sqrt(p * q), dim=0)
        BC_N = - torch.log(spq_N)

        # Try to 'focus' the loss on cases where p is different from q
        #             similar  dissimilar
        # spq          0.99       0.01
        # BC           0.01       4.6
        # (1 - spq)    0.01       0.99
        # focussed     0.0001     4.6
        # BC_N = - (1 - spq_N) * torch.log(spq_N)
        # NOTE: this does not seem to do much

        return torch.sum(BC_N)

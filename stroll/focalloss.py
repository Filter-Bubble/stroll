# Taken from https://github.com/clcarwin/focal_loss_pytorch.github
# License: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# https://en.wikipedia.org/wiki/Bhattacharyya_distance
class Bhattacharyya(nn.Module):
    def __init__(self):
        super(Bhattacharyya, self).__init__()
        d = torch.eye(21)

        # Labels 0 - 4 are Arg[0-5], and are similar
        d[0:6, 0:6] += 0.01

        # Labels 5 - 19 are ArgM, and are similar
        d[6:20, 6:20] += 0.01

        self.q = F.softmax(d, dim=0)

    def forward(self, input, target):
        """Input shape [C,N], target shape [N]"""

        q = self.q[target.long()].transpose(0, 1)
        p = F.softmax(input, dim=0)
        BC_N = torch.sum(torch.sqrt(p * q), dim=0)
        BC_N = - torch.log(BC_N)

        return torch.sum(BC_N)

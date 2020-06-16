import torch

import sys
import math
import numpy as np

from torch.utils.data import RandomSampler


def get_optimizer_and_scheduler_for_net(
        net,
        solver='CE',
        learning_rate=1e-2,
        ):
    if solver == 'SGD':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            momentum=0.9
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.9
            )
    elif solver == 'ADAM':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate,
            )
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=[lambda epoch: 0.9**min(max(0, (epoch - 4) // 2), 36)],
        #     )
    elif solver == 'ADAMW':
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=learning_rate
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=1.0
            )
    else:
        print('Solver not implemented.')
        sys.exit(-1)

    return optimizer, scheduler


class RandomBatchSampler:
    """
    Randomly sample batches; but keep elements within a batch consecutive.
    """
    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size

        # find the number of batches
        self.nbatches = math.ceil(length / batch_size)

        # create a random sampler over these batches
        self.random_sampler = RandomSampler(range(self.nbatches))

    def __len__(self):
        return self.nbatches

    def __iter__(self):
        self.it = self.random_sampler.__iter__()
        return self

    def __next__(self):
        batch = next(self.it)

        start = batch * self.batch_size
        end = min(start + self.batch_size, self.length)
        return range(start, end)


class RandomDocSampler:
    """
    Randomly sample single documents from a ConlluDataset
    """
    def __init__(self, dataset):
        self.ndocs = len(dataset.doc_lengths)
        self.shuffled_docs = []

        doc_lengths = dataset.doc_lengths
        self.doc_lengths = [doc_lengths[i] for i in doc_lengths]

        self.doc_start = np.zeros(self.ndocs, dtype=np.int32)
        self.doc_end = np.zeros(self.ndocs, dtype=np.int32)
        cumsum = 0
        for i in range(self.ndocs):
            self.doc_start[i] = cumsum
            self.doc_end[i] = cumsum + self.doc_lengths[i]
            cumsum += self.doc_lengths[i]

        # create a random sampler over the documents
        self.random_sampler = RandomSampler(range(self.ndocs))

    def __iter__(self):
        self.shuffled_docs = list(self.random_sampler)
        return self

    def __len__(self):
        return self.ndocs

    def __next__(self):
        if len(self.shuffled_docs) == 0:
            raise StopIteration

        doc = self.shuffled_docs.pop()
        sentences = range(self.doc_start[doc], self.doc_end[doc])

        return sentences

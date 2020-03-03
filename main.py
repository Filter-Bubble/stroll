import time
import signal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter

import torchvision

import dgl
import dgl.function as fn

from stroll.graph import GraphDataset, draw_graph
from stroll.model import Net
from stroll.labels import BertEncoder

torch.manual_seed(43)

# Not used at the moment
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

if __name__ == '__main__':
    # Skip loading bert for now (this is a bit slow)
    # sentence_encoder = BertEncoder()
    sentence_encoder = None

    sonar = GraphDataset('sonar1_fixed.conllu', sentence_encoder=sentence_encoder)
    train_length = int(0.9 * len(sonar))
    test_length = len(sonar) - train_length
    train_set, test_set = random_split(sonar, [train_length, test_length])

    # Test setings
    test_graph = dgl.batch([g for g in test_set])
    print ('Test set contains {} words.'.format(len(test_graph)))
    # Create network
    # out_feats:
    # ROLE := 21
    # FRAME := 2
    net = Net(in_feats=sonar.in_feats, h_dims=16, out_feats=21)
    print(net)

    def sigterm_handler(_signo, _stack_frame):
        print('Aborting')
        writer.close()
        torch.save(net.state_dict(), './model.pt')
        exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    # Training settings
    #  * mini batch size of 10 sentences
    #  * shuffle the data on each epochs
    trainloader = DataLoader(train_set, batch_size=10, shuffle=True, collate_fn=dgl.batch)
    #  * Adam with fixed learning rate fo 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    #  * 2 epochs
    num_epochs = 2

    # Log settings
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/experiment1')

    # diagnostic settings
    word_count = 0
    count_per_eval = 1000
    next_eval = count_per_eval
    t0 = time.time()

    # loop over the epochs
    for epoch in range(num_epochs):
        # loop over each minibatch
        for g in trainloader:
            net.train()

            # inputs (minibatch, C, d_1, d_2, ..., d_K)
            # target (minibatch,    d_1, d_2, ..., d_K)
            # minibatch = minibatch Size
            # C         = number of classes
            # d_x       = extra dimensions
            logits = net(g).transpose(0,1)
            target = g.ndata['role']
            loss = F.cross_entropy(logits.view(1,21,-1), target.view(1,-1))

            # apply loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # diagnostics
            word_count = word_count + len(g)

            if word_count > next_eval:
                # draw_graph(g)

                dur = time.time() - t0
                acc1 = net.evaluate_frame(test_graph)
                acc2 = net.evaluate_role(test_graph)
                print("Elements {:08d} | Loss {:.4f} | Acc1 {:.4f} | Acc2 {:.4f} | words/sec {:4.3f}".format(
                        word_count, loss.item(), acc1, acc2, len(g) / dur
                        )
                     )

                grid = torchvision.utils.make_grid([torch.eye(2)])
                writer.add_image('confusion_matrix-Frame', grid)

                grid = torchvision.utils.make_grid([torch.eye(16)])
                writer.add_image('confusion_matrix-Role', grid)

                writer.add_scalar('training loss', loss.item(), word_count)
                writer.add_scalar('accuracy_frame', acc1, word_count)
                writer.add_scalar('accuracy_role', acc2, word_count)

                for name, param in net.state_dict().items():
                     writer.add_histogram('hist_' + name, param, word_count)
                     writer.add_scalar('norm_' + name, torch.norm(param), word_count)

                # reset timer
                next_eval = next_eval + count_per_eval
                t0 = time.time()

        print ('Epoch {} done'.format(epoch))

    torch.save(net.state_dict(), './model.pt')
    writer.close()

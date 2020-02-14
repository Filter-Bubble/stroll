import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

import time
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

class BertEncoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-dutch-cased")
        self.model = BertModel.from_pretrained("bert-base-dutch-cased")
        self.model.eval()

    def __call__(self, sentence):
        # TODO: [CLS] and [SEP] symbols?
        text = " ".join(sentence)

        # Tokenize input
        tokenized_text = self.tokenizer.tokenize(text)

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
             outputs = self.model(tokens_tensor)

        # torch tensor of shape (batch_size=1, sequence_length, hidden_size=768)
        hidden_states = outputs[0]

        outputs = []
        # Realign the tokens
        # ['Repareer',             'de', 'tokenizatie',                   '.']
        # ['Rep', '##are', '##er', 'de', 'to', '##ken', '##iza', '##tie', '.']
        bert_i = 0
        while len(outputs) < len(sentence):
            # skip partial tokens
            while tokenized_text[bert_i][0] == '#':
                bert_i = bert_i + 1
            outputs.append(hidden_states[0, bert_i])
            bert_i = bert_i + 1
        
        return outputs


class SonarDataset(Dataset):
    def _init_labelencoders(self):
        self.upos_codec.fit([
            '_', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
            ])
        self.deprel_codec.fit([
            '_', 'acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos',
            'aux', 'aux:pass', 'case', 'cc', 'ccomp', 'compound:prt', 'conj',
            'cop', 'csubj', 'det', 'expl', 'expl:pv', 'fixed', 'flat', 'iobj',
            'mark', 'nmod', 'nmod:poss', 'nsubj', 'nsubj:pass', 'nummod',
            'obj', 'obl', 'obl:agent', 'orphan', 'parataxis', 'punct', 'root', 'xcomp'
            ])
        self.frame_codec.fit([
            '_', 'rel'
            ])
        self.role_codec.fit([
            '_',
            'Arg0', 'Arg1', 'Arg2', 'Arg3', 'Arg4', 'Arg5', 'ArgM-ADV', 'ArgM-CAU',
            'ArgM-DIR', 'ArgM-DIS', 'ArgM-EXT', 'ArgM-LOC', 'ArgM-MNR', 'ArgM-MOD',
            'ArgM-NEG', 'ArgM-PNC', 'ArgM-PRD', 'ArgM-REC', 'ArgM-STR', 'ArgM-TMP'
            ])
        self.feats_codec.fit([
            '_', 'Abbr=Yes', 'Case=Acc', 'Case=Dat', 'Case=Gen', 'Case=Nom', 'Definite=Def',
            'Definite=Ind', 'Degree=Cmp', 'Degree=Pos', 'Degree=Sup', 'Foreign=Yes',
            'Gender=Com', 'Gender=Com,Neut', 'Gender=Neut', 'Number=Plur', 'Number=Sing',
            'Person=1', 'Person=2', 'Person=3', 'PronType=Dem', 'PronType=Ind', 'PronType=Int',
            'PronType=Prs', 'PronType=Rcp', 'PronType=Rel', 'Reflex=Yes', 'Tense=Past',
            'Tense=Pres', 'VerbForm=Fin', 'VerbForm=Inf', 'VerbForm=Part'
            ])

    def _load_conllu(self, filename):
        logging.info("Opening {}".format(filename))

        with open(filename, "r") as f:
            conllu_raw = f.readlines()

        # sent_id
        # text
        # <12 columns tab separated, 1 line per token in the sentence>
        # <empty line>
        sentence = [] 
        for line in conllu_raw:
            # ignore comments
            # TODO: sent_id
            if line[0] == '#':
                continue

            # remove possible trailing newline and whitespace
            fields = line.rstrip().split('\t')
            if len(fields) == 1 and fields[0] == '':

                # newline means end of a sentence
                if len(sentence) > 0:

                    raw_sentence = []
                    for token in sentence:
                        raw_sentence.append(token[self.FORM])

                    # add the finished sentence to dataset
                    self.sentences.append(sentence)
                    self.raw_sentences.append(raw_sentence)

                # start a new sentence
                sentence = []    
            elif len(fields) == 12:
                if fields[self.ID].find('.') == -1:
                    # convert the strings to integers for the relevant fields
                    fields[self.ID] = int(fields[self.ID])
                    fields[self.HEAD] = int(fields[self.HEAD])
                    sentence.append(fields)
                else:
                    # this is an extra subtoken, fi. '14.1', drop it
                    pass
            else:
                logging.warn('Incorrect number of fields in sentence: {} "{}"'.format(len(fields), line))

    def __init__(self, filename, sentence_encoder=None):
        self.sentences = []
        self.raw_sentences = []
        self.ID = 0
        self.FORM = 1
        self.LEMMA = 2
        self.UPOS = 3
        self.XPOS = 4
        self.FEATS = 5
        self.HEAD = 6
        self.DEPREL = 7
        self.DEPS = 8
        self.MISC = 9
        self.FRAME = 10
        self.ROLE = 11

        self.upos_codec = LabelEncoder()
        self.deprel_codec = LabelEncoder()
        self.frame_codec= LabelEncoder()
        self.role_codec = LabelEncoder()
        self.feats_codec = LabelEncoder()

        self.sentence_encoder = sentence_encoder

        self._init_labelencoders()
        self._load_conllu(filename)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        raw_sentence = self.raw_sentences[index]

        itm = []

        # Encode the FORMS using a sentence encoder
        if (self.sentence_encoder):
            encoded_sentence = self.sentence_encoder(raw_sentence)

        # First add the simple, one-hot-encoded features
        for i, word in enumerate(sentence):
            # INPUTS

            # grah structure, including 'form' for drawing the graph
            wid = word[self.ID]
            form = string_to_tensor(word[self.FORM], 20).view(1,20)
            head = word[self.HEAD]

            # features
            # FEATS is a '|' separated list of labels
            # we'll get a 2D tensor; collapse (sum) it to a vector
            upos = self.to_one_hot(self.upos_codec, [word[self.UPOS]])
            deprel = self.to_one_hot(self.deprel_codec, [word[self.DEPREL]])
            feats = self.to_one_hot(self.feats_codec, word[self.FEATS].split('|')).sum(axis=0, keepdim=True)

            if self.sentence_encoder:
                embedding = encoded_sentence[i]
            else:
                embedding = torch.zeros([1,1])

            # OUTPUTS
            frame = torch.tensor(self.frame_codec.transform([word[self.FRAME]]))
            role = torch.tensor(self.role_codec.transform([word[self.ROLE]]))

            newitem = (wid, form, upos, feats, head, deprel, embedding, frame, role)
            itm.append( newitem )

        return itm

    def __len__(self):
        return len(self.sentences)

    def to_one_hot(self, codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_))[value_idxs]


def string_to_tensor(s, max_length=16):
    t = torch.zeros(max_length)

    # convert to bytearray
    b = s.encode()

    # first number is the string length (truncated)
    l = min(len(b), max_length - 1)

    t[0] = l
    for i in range(l):
        t[i+1] = b[i]

    return t

def tensor_to_string(tt):
    # cast to 1d tensor
    t = tt.view(-1)

    # get the string length
    l = int(t[0])

    # cast to a byte array
    s = bytearray([ int(c) for c in t[1: l+1 ].tolist() ])

    return s.decode()

def sentence_to_graph(sentence):
    g = dgl.DGLGraph()

    wid_to_nid = {}

    # add nodes
    for token in sentence:
        wid, form, upos, feats, head, deprel, embedding, frame, role = token
        g.add_nodes(1, {
            'form': form,
            'upos': upos,
            'feats': feats,
            'embedding': embedding,
            'frame': frame,
            'role': role
            })

        wid_to_nid[wid] = len(g) - 1

    # add edges
    for token in sentence:
        wid, form, upos, feats, head, deprel, embedding, frame, role = token
        if head != 0:
            g.add_edges(wid_to_nid[wid], wid_to_nid[head], {
                'deprel': deprel
                })

    return g

def draw_graph(graph):
    label_tensor = graph.ndata['form']

    ng = graph.to_networkx()
    nx.relabel_nodes(ng,
            lambda x: tensor_to_string(label_tensor[x]),
            copy=False)

    nx.draw(ng, with_labels=True)
    plt.show()


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        feats = self.linear(node.data['h'])
        if self.activation is not None:
            feats = self.activation(feats)
        return {'h' : feats}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):

        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.sum(msg='m', out='h')

        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(17, 16, F.relu)
        self.gcn2 = GCN(16, 2, None)

    def forward(self, g):
        x = self.gcn1(g, g.ndata['upos'])
        x = self.gcn2(g, x)
        return x

def evaluate(model, g, p=False):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        labels = g.ndata['frame']
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

if __name__ == '__main__':
    # sentence_encoder = BertEncoder()
    # mysonar = SonarDataset('sonar1_fixed.conllu', sentence_encoder=sentence_encoder)
    mysonar = SonarDataset('sonar1_fixed.conllu') # Skip loading bert for now (this is much faster)

    g = sentence_to_graph(mysonar[700])

    # add self edges
    g.add_edges(g.nodes(), g.nodes(), {
        'deprel': mysonar.to_one_hot(mysonar.deprel_codec, ['_']) # TODO: separate symbol for self edges?
        })

    # print (mysonar.raw_sentences[700])
    draw_graph(g)

    # Create the model using the graph
    net = Net()
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    dur = []
    for epoch in range(400):
        if epoch >=3:
            t0 = time.time()

        net.train()
        logits = net(g).transpose(0,1)
        target = g.ndata['frame']

        # inputs (minibatch, C, d_1, d_2, ..., d_K) -> torch.Size([2, 7])
        # target (minibatch,    d_1, d_2, ..., d_K) -> tensor([0, 0, 0, 0, 0, 0, 0])
        # minibatch = minibatch Size
        # C         = number of classes
        # d_x       = extra dimensions
        loss = F.cross_entropy(logits.view(1,2,8), target.view(1,8))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)

        acc = evaluate(net, g)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))

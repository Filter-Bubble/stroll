import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import fasttext


UPOS = [
        '_', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
        'NUM', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
        ]

XPOS = [
        '_', '1', '2', '2b', '2v', '3', '3m', '3o', '3p', '3v', 'aanw', 'ADJ',
        'adv-pron', 'afgebr', 'afk', 'agr', 'basis', 'bep', 'betr', 'bez',
        'bijz', 'BW', 'comp', 'conj', 'dat', 'deeleigen', 'det', 'dim',
        'eigen', 'enof', 'ev', 'evf', 'evmo', 'evon', 'evz', 'excl', 'fem',
        'fin', 'gen', 'genus', 'getal', 'grad', 'hoofd', 'inf', 'init',
        'LET', 'LID', 'masc', 'meta', 'met-e', 'met-s', 'met-t', 'mv',
        'mv-n', 'N', 'nadr', 'neven', 'nom', 'nomin', 'obl', 'od', 'onbep',
        'onder', 'onz', 'pers', 'persoon', 'postnom', 'pr', 'prenom',
        'pron', 'pv', 'rang', 'recip', 'red', 'refl', 'rest', 'rest3',
        'soort', 'SPEC', 'stan', 'sup', 'symb', 'tgw', 'TSW', 'TW', 'vb',
        'vd', 'verl', 'versm', 'VG', 'VNW', 'vol', 'vreemd', 'vrij', 'VZ',
        'WW', 'zijd', 'zonder', 'zonder-n'
        ]

DEPREL = [
        '_', 'acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos',
        'aux', 'aux:pass', 'case', 'cc', 'ccomp', 'compound:prt', 'conj',
        'cop', 'csubj', 'det', 'expl', 'expl:pv', 'fixed', 'flat', 'iobj',
        'mark', 'nmod', 'nmod:poss', 'nsubj', 'nsubj:pass', 'nummod', 'obj',
        'obl', 'obl:agent', 'orphan', 'parataxis', 'punct', 'root', 'xcomp'
        ]

FEATS = [
        '_', 'Abbr=Yes', 'Case=Acc', 'Case=Dat', 'Case=Gen', 'Case=Nom',
        'Definite=Def', 'Definite=Ind', 'Degree=Cmp', 'Degree=Pos',
        'Degree=Sup', 'Foreign=Yes', 'Gender=Com', 'Gender=Com,Neut',
        'Gender=Neut', 'Number=Plur', 'Number=Sing', 'Person=1', 'Person=2',
        'Person=3', 'PronType=Dem', 'PronType=Ind', 'PronType=Int',
        'PronType=Prs', 'PronType=Rcp', 'PronType=Rel', 'Reflex=Yes',
        'Tense=Past', 'Tense=Pres', 'VerbForm=Fin', 'VerbForm=Inf',
        'VerbForm=Part'
        ]

# NOTE: the alphabetical ordering is important to keep correct weights
ROLES = [
        'Arg0', 'Arg1', 'Arg2', 'Arg3', 'Arg4', 'Arg5', 'ArgM-ADV', 'ArgM-CAU',
        'ArgM-DIR', 'ArgM-DIS', 'ArgM-EXT', 'ArgM-LOC', 'ArgM-MNR', 'ArgM-MOD',
        'ArgM-NEG', 'ArgM-PNC', 'ArgM-PRD', 'ArgM-REC', 'ArgM-STR', 'ArgM-TMP',
        '_'
        ]

ROLE_WEIGHTS = torch.Tensor([ # 3652
     0.500,  # 0.5,  # Arg0              18026,
     0.291,  # 0.5,  # Arg1              30935,
     1.298,  # 1.5,  # Arg2              6944,    1.5   1.5 2
     9.000,  # 2.0,  # Arg3              502,
     9.000,  # 2.0,  # Arg4              594,
     0.001,  # 2.0,  # Arg5              2,
     1.801,  # 2.0,  # ArgM-ADV          5005,
     5.811,  # 2.0,  # ArgM-CAU          1551,
     9.000,  # 2.0,  # ArgM-DIR          548,
     1.711,  # 2.0,  # ArgM-DIS          5267,
     9.000,  # 2.0,  # ArgM-EXT          912,
     1.354,  # 1.8,  # ArgM-LOC          6657,    1.8   2   2
     1.866,  # 1.8,  # ArgM-MNR          4831,    1.8   2   2
     1.549,  # 2.0,  # ArgM-MOD          5818,
     3.120,  # 2.0,  # ArgM-NEG          2889,
     5.112,  # 2.0,  # ArgM-PNC          1763,
     7.892,  # 2.0,  # ArgM-PRD          1142,
     7.645,  # 2.0,  # ArgM-REC          1179,
     0.001,  # 2.0,  # ArgM-STR          5,
     0.907,  # 2.0,  # ArgM-TMP          9939,
     0.023,  # 2e-2,  # _                386898   2e-3 1e-3 1e-4
     ])
# ROLE_WEIGHTS = [
#     2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
#     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
#     1e-4,
#     ]
# ROLE_COUNTS = torch.Tensor([
#    18026, 30935, 6944, 502, 594, 2,
#    5005, 1551, 548, 5267, 912, 6657, 4831,
#    5818, 2889, 1763, 1142, 1179, 5, 9939,
#    386898
#    ])
# ROLE_WEIGHTS = 1.0 / ROLE_COUNTS
# ROLE_WEIGHTS = 21.0 * ROLE_WEIGHTS / ROLE_WEIGHTS.sum()

# NOTE: the alphabetical ordering is important to keep correct weights
FRAMES = ['_', 'rel']
# FRAME_COUNTS = torch.Tensor([454921, 36486])
# FRAME_WEIGHTS = 1.0 / FRAME_COUNTS
# FRAME_WEIGHTS = FRAME_WEIGHTS / FRAME_WEIGHTS.sum()
FRAME_WEIGHTS = torch.Tensor([1., 10.])

upos_codec = LabelEncoder().fit(UPOS)
xpos_codec = LabelEncoder().fit(XPOS)
deprel_codec = LabelEncoder().fit(DEPREL)
feats_codec = LabelEncoder().fit(FEATS)
frame_codec = LabelEncoder().fit(FRAMES)
role_codec = LabelEncoder().fit(ROLES)


def to_one_hot(codec, values):
    if type(values) == type([]):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_))[value_idxs].sum(axis=0)
    else:
        value_idxs = codec.transform([values])
        return torch.eye(len(codec.classes_))[value_idxs].flatten()


def to_index(codec, values):
    return torch.tensor(codec.transform([values])).flatten()


class FasttextEncoder:
    """Use Fasttext word vectors per word"""
    def __init__(self, filename):
        self.model = fasttext.load_model(filename)
        self.dims = self.model.get_dimension()
        self.name = 'FT{}'.format(self.dims)

    def __call__(self, sentence):
        word_vectors = []
        for token in sentence:
            word_vectors.append(torch.Tensor(self.model[token.FORM]))

        return word_vectors


class BertEncoder:
    """Use averaged bert vectors over constituent tokens"""
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-dutch-cased")
        self.model = BertModel.from_pretrained("bert-base-dutch-cased")
        self.model.eval()
        self.dims = 768
        self.name = 'BERT'

    def __call__(self, sentence):
        self.model.eval()

        # Tokenize input
        # TODO: [CLS] and [SEP] symbols?
        tokenized_text = self.tokenizer.tokenize(sentence.full_text)

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            # torch tensor of shape
            # (batch_size=1, sequence_length, hidden_size=768)
            bert_output, _ = self.model(tokens_tensor)

        # Realign the tokens and build the word vectors by averaging
        # ['Repareer',             'de', 'token-izatie',                   ]
        # ['Rep', '##are', '##er', 'de', 'to', '##ken', '-', 'iza', '##tie']
        word_vectors = []

        # Align tokenizations
        #  * BERT can break up our 'gold' tokens in smaller parts
        #  * BERT does not merge any of our white-space separated tokens
        bert_i = 0
        # loop over the gold tokens
        for gold_i, gold_t in enumerate(sentence):
            chars_gold = len(gold_t.FORM)
            chars_bert = 0
            subword_tensors = []

            # keep adding BERT tokens until they make up the gold token
            while chars_bert < chars_gold and bert_i < len(tokenized_text):
                subword_tensors.append(bert_output[0, bert_i])

                bert_t = tokenized_text[bert_i]
                if bert_t == self.tokenizer.unk_token:
                    # assume the unk token stands for the whole remaining gold
                    # token
                    chars_bert = chars_gold
                else:
                    chars_bert = chars_bert + len(bert_t)
                    if bert_t[0:2] == '##':
                        chars_bert = chars_bert - 2

                bert_i = bert_i + 1

            # average and append to OUtput
            try:
                word_vectors.append(
                        torch.mean(torch.stack(subword_tensors, dim=1), 1)
                        )
            except:
                # print (sentence)
                # print ('len subword_tensors', len(subword_tensors))
                # print (bert_i, chars_bert, gold_i, chars_gold, gold_t)
                word_vectors.append(torch.zeros([self.dims]))

        return word_vectors

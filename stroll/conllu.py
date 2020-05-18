import logging
import torch
from torch.utils.data import Dataset
from .labels import upos_codec, xpos_codec, deprel_codec, feats_codec, \
        frame_codec, role_codec
from .labels import to_one_hot, to_index
from .labels import ROLES, FRAMES


class Token():
    """A class representing a single token, ie. a word, with its annotation."""
    def __init__(self, fields, isEncoded=False):
        self.isEncoded = isEncoded
        if len(fields) < 10:
            logging.warn(
               'Incorrect number of fields in sentence: {}'.format(len(fields))
               )
        else:
            self.ID = fields[0]
            self.FORM = fields[1]
            self.LEMMA = fields[2]
            self.UPOS = fields[3]
            self.XPOS = fields[4]
            self.FEATS = fields[5]
            self.HEAD = fields[6]
            self.DEPREL = fields[7]
            self.DEPS = fields[8]
            self.MISC = fields[9]

        # Treat fields 10 and 11 as frame and role
        # NOTE: this a private extension the to conllu format
        if len(fields) >= 12:
            self.FRAME = fields[10]
            self.ROLE = fields[11]
            self.pFRAME = 1.
            self.pROLE = 1.
        else:
            self.FRAME = '_'
            self.ROLE = '_'
            self.pFRAME = 0.
            self.pROLE = 0.

        # Treat field 12 as co-reference info
        # NOTE: this a private extension the to conllu format
        if len(fields) >= 13:
            self.COREF = fields[12]
        else:
            self.COREF = '_'

        # We also allow labelling using sentence encoders (BERT/ FastText)
        self.WVEC = None

    def __repr__(self):
        if self.isEncoded:
            return 'Encoded'
        else:
            # Used for outputting back to conllu
            return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                self.ID, self.FORM, self.LEMMA, self.UPOS, self.XPOS,
                self.FEATS, self.HEAD, self.DEPREL, self.DEPS, self.MISC,
                self.FRAME, self.ROLE, self.COREF
                )

    def __getitem__(self, index):
        if index == 'UPOS':
            return self.UPOS
        elif index == 'XPOS':
            return self.XPOS
        elif index == 'FEATS':
            return self.FEATS
        elif index == 'DEPREL':
            return self.DEPREL
        elif index == 'WVEC':
            return self.WVEC
        elif index == 'RID':
            return self.RID
        elif index == 'COREF':
            return self.COREF
        return None

    def encode(self):
        if self.COREF == '_':
            coref = torch.tensor([-1], dtype=torch.int32)
        else:
            coref = torch.tensor([int(self.COREF)], dtype=torch.int32)

        return Token([
            self.ID,  # not encoded
            self.FORM,  # encoded later by sentence encoder
            self.LEMMA,  # not encoded
            to_one_hot(upos_codec, self.UPOS),
            to_one_hot(xpos_codec, self.XPOS.split('|')),
            to_one_hot(feats_codec, self.FEATS.split('|')),
            self.HEAD,  # not encoded
            to_one_hot(deprel_codec, self.DEPREL),
            self.DEPS,  # not encoded
            self.MISC,  # not encoded
            to_index(frame_codec, self.FRAME),
            to_index(role_codec, self.ROLE),
            coref
            ], isEncoded=True)


class Sentence():
    """A class representing a sentence,
    ie. the tokens with their annotations."""
    def __init__(self, sent_id=None, full_text=None):
        self.sent_id = sent_id  # sentence identifier, string
        self.full_text = full_text  # full (raw) text of sentence
        self.doc_id = None
        self.tokens = []
        self._id_to_index = None  # maps Token.ID to int index in sentence

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return '# sent_id = ' + self.sent_id + '\n' + \
               '# text = ' + self.full_text + '\n' + \
               '\n'.join([token.__repr__() for token in self.tokens])

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.index(index)
        return self.tokens[index]

    def index(self, ID):
        if self._id_to_index is None:
            self._build_id_to_index()
        return self._id_to_index[ID]

    def __iter__(self):
        for i in range(len(self.tokens)):
            yield self.tokens[i]

    def _build_id_to_index(self):
        self._id_to_index = {}
        for i, token in enumerate(self.tokens):
            self._id_to_index[token.ID] = i

    def add(self, token):
        # TODO: see if we can keep those tokens.
        if token.ID.find('.') == -1:
            self.tokens.append(token)

        # force rebuilding of the ID lookup table and adjacency matrix
        self._id_to_index = None

    def set_full_text(self, full_text):
        self.full_text = full_text

    def set_sent_id(self, sent_id):
        self.sent_id = sent_id

    def encode(self, sentence_encoder=None):
        encoded_sentence = Sentence()
        encoded_sentence.set_sent_id(self.sent_id)
        encoded_sentence.set_full_text(self.full_text)

        for token in self.tokens:
            encoded_sentence.add(token.encode())

        # Encode the FORMS using a sentence encoder
        if sentence_encoder:
            for i, WVEC in enumerate(sentence_encoder(self)):
                encoded_sentence.tokens[i].WVEC = WVEC

        return encoded_sentence


class ConlluDataset(Dataset):
    def __init__(self, filename=None):
        self.sentences = []

        if filename is not None:
            self._load(filename)

    def _load(self, filename):
        logging.info("Opening {}".format(filename))

        with open(filename, "r") as f:
            conllu_raw = f.readlines()

        # sent_id = 116
        # text = Wie kan optreden ?
        # <12 columns tab separated, 1 line per token in the sentence>
        # <empty line>
        sentence = Sentence()
        doc_count = 0
        doc_current_id = filename
        for line in conllu_raw:

            # remove possible trailing newline and whitespace
            line = line.rstrip()

            if line[0:12] == '# sent_id = ':
                # store sentence id
                sentence.set_sent_id(line[12:])
                continue
            elif line[0:9] == '# text = ':
                # store sentence full text
                sentence.set_full_text(line[9:])
                continue
            elif line[0:8] == '# newdoc':
                # store doc id if present: '# newdoc id = mf920901-001'
                if len(line) > 14:
                    doc_current_id = line[14:]
                else:
                    doc_current_id = '{filename}-{:06d}'.format(
                            filename, doc_count)
                doc_count += 1
            elif line[0:1] == '#':
                # ignore comments
                continue
            elif len(line) == 0:
                # newline means end of a sentence
                if len(sentence) > 0:
                    self.index = len(self.sentences)
                    sentence.doc_id = doc_current_id
                    self.sentences.append(sentence)

                # start a new sentence
                sentence = Sentence()
            else:
                fields = line.split('\t')
                sentence.add(Token(fields))

    def __getitem__(self, index):
        return self.sentences[index]

    def __iter__(self):
        for i in range(len(self.sentences)):
            yield self.sentences[i]

    def __len__(self):
        return len(self.sentences)

    def statistics(self):
        role_counts = {}
        for r in ROLES:
            role_counts[r] = 0

        frame_counts = {}
        for f in FRAMES:
            frame_counts[f] = 0

        for sentence in self.sentences:
            for token in sentence:
                role_counts[token.ROLE] = role_counts[token.ROLE] + 1
                frame_counts[token.FRAME] = frame_counts[token.FRAME] + 1

        return role_counts, frame_counts

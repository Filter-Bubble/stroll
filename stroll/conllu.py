import logging
from torch.utils.data import Dataset
from .labels import upos_codec, xpos_codec, deprel_codec, feats_codec, \
        frame_codec, role_codec
from .labels import to_one_hot, to_index
from .labels import ROLES, FRAMES


def is_ok(sentence):
    # Check if the syntactic head of each argument is also a SRL Frame
    ID_to_frame = {}
    ID_to_frame['0'] = '_'  # for when looking at the frame of the head
    for idx, token in enumerate(sentence):
        ID_to_frame[token.ID] = token.FRAME
    for token in sentence:
        if token.ROLE != '_':
            if ID_to_frame[token.HEAD] != 'rel':
                print('Dropping:', sentence.sent_id)
                return False
    return True


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
        if len(fields) > 9:
            self.FRAME = fields[10]
            self.ROLE = fields[11]
        else:
            self.FRAME = '_'
            self.ROLE = '_'

        # We also allow labelling using sentence encoders (BERT)
        self.WVEC = None

    def __repr__(self):
        if self.isEncoded:
            return 'Encoded'
        else:
            # Used for outputting back to conllu
            return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                self.ID, self.FORM, self.LEMMA, self.UPOS, self.XPOS,
                self.FEATS, self.HEAD, self.DEPREL, self.DEPS, self.MISC,
                self.FRAME, self.ROLE
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
        return None

    def encode(self):
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
            ], isEncoded=True)


class Sentence():
    """A class representing a sentence,
    ie. the tokens with their annotations."""
    def __init__(self, sent_id=None, full_text=None):
        self.sent_id = sent_id
        self.full_text = full_text
        self.tokens = []

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return \
                '# sent_id = ' + self.sent_id + '\n' + \
                '# text = ' + self.full_text + '\n' + \
                '\n'.join([token.__repr__() for token in self.tokens])

    def __getitem__(self, index):
        return self.tokens[index]

    def __iter__(self):
        for i in range(len(self.tokens)):
            yield self.tokens[i]

    def add(self, token):
        if token.ID.find('.') == -1:
            self.tokens.append(token)

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
    def __init__(self, filename):
        self.sentences = []

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
            elif line[0:1] == '#':
                # ignore comments
                continue
            elif len(line) == 0:
                # newline means end of a sentence
                if len(sentence) > 0:
                    self.sentences.append(sentence)
                    # add the finished sentence to the dataset
                    # if is_ok(sentence):
                    #     self.sentences.append(sentence)

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

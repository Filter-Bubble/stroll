import logging
import torch
from torch.utils.data import Dataset
from .labels import upos_codec, xpos_codec, deprel_codec, feats_codec, \
        frame_codec, role_codec
from .labels import to_one_hot, to_index
from .labels import ROLES, FRAMES
from collections import OrderedDict
import os


COPULA_NOUN_DESC_MOVE_TO_VERB = [
        'advcl', 'advmod', 'aux', 'aux:pass', 'case', 'cc', 'csubj', 'expl',
        'expl:pv', 'iobj', 'mark', 'nsubj', 'obl', 'obl:agent', 'orphan',
        'parataxis', 'punct'
        ]


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

        # For coreference resolution
        if len(fields) >= 14:
            self.anaphore = float(fields[13])
        else:
            self.anaphore = -100

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
        elif index == 'FRAME':
            return self.FRAME
        elif index == 'ROLE':
            return self.ROLE
        elif index == 'WVEC':
            return self.WVEC
        elif index == 'COREF':
            return self.COREF
        return None

    def encode(self):
        if self.COREF in ['_', '-']:
            coref = torch.tensor([0], dtype=torch.int32)
        else:
            coref = torch.tensor([int(self.COREF) + 1], dtype=torch.int32)

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
            coref,
            self.anaphore
            ], isEncoded=True)


class Sentence():
    """
    A class representing a sentence.

    It contains the tokens with their annotations.
    Like a dataset, it is iterable and index-able:
        for token in sentence
        token = Sentence[1]  # indexed by number
        token = Sentence['0']  # indexed by Token.ID

    NOTE: You can use id = sentence.index['0'] to get the
    number corresponding to the token's Token.ID

    Properties:
        sent_id    identifier of the sentence
        full_text  full text of the sentence, possibly not tokenized.
        doc_id     identifier of the document this sentence is from.
        doc_rank   the document's rank (first, second, ..) in the dataset
        sent_rank  the sentence's rank (first, second, ..) in the document
        tokens     the list of tokens that make up the sentence
    """
    def __init__(self,
                 sent_id=None,
                 full_text=None
                 ):
        self.sent_id = sent_id  # sentence identifier, string
        self.full_text = full_text  # full (raw) text of sentence
        self.sent_rank = None
        self.doc_rank = None
        self.doc_id = None
        self.dataset = None
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
        encoded_sentence = Sentence(
                sent_id=self.sent_id,
                full_text=self.full_text
                )
        encoded_sentence.sent_rank = self.sent_rank
        encoded_sentence.doc_rank = self.doc_rank
        encoded_sentence.doc_id = self.doc_id
        encoded_sentence.dataset = self.dataset

        for token in self.tokens:
            encoded_sentence.add(token.encode())

        # Encode the FORMS using a sentence encoder
        if sentence_encoder:
            for i, WVEC in enumerate(sentence_encoder(self)):
                encoded_sentence.tokens[i].WVEC = WVEC

        return encoded_sentence


class ConlluDataset(Dataset):
    """
    The conll-u dataset class.

    The dataset is index-able, and iterable:
        sent = Dataset[10]
        for sent in Dataset

    properties:
        sentences    list of Sentence
        doc_lengths  dict of number of sentences per doc,
                     indexed by Sentenc.doc_id
    """
    def __init__(self, filename=None):
        self.sentences = []
        self.doc_lengths = OrderedDict()

        if filename is not None:
            self._load(filename)

    def __repr__(self):
        res = ''
        current_doc_id = ''
        first = True
        for sentence in self:
            if not first:
                res += '\n'

            # keep track of the current document
            if sentence.doc_id != current_doc_id:
                current_doc_id = sentence.doc_id
                if not first:
                    res += '\n'
                res += '# newdoc id = {}\n'.format(sentence.doc_id)
            res += sentence.__repr__()
            res += '\n'

            if first:
                first = False
        return res

    def load_conll2012(self, filename):
        logging.info("Opening {}".format(filename))

        with open(filename, "r") as f:
            conll2012_raw = f.readlines()

        sent_rank = 1
        full_text = []
        sentence = Sentence()
        doc_current_id = filename  # use the filename as default doc_id
        for line in conll2012_raw:

            # remove possible trailing newline and whitespace
            line = line.rstrip()

            # #begin document (dpc-bmm-001086-nl-sen); part 000
            if line[0:15] == '#begin document':
                doc_current_id = line[17:]
                try:
                    end = doc_current_id.index(')')
                    doc_current_id = doc_current_id[:end]
                except ValueError:
                    pass

            # #end document
            elif line[0:13] == '#end document':
                doc_current_id = filename  # use the filename as default doc_id
                sent_rank = 1

            # <newline> is sentence separator
            elif len(line) == 0:
                if len(sentence) > 0:
                    sentence.doc_id = doc_current_id
                    sentence.sent_id = '{}'.format(sent_rank)
                    sentence.full_text = ' '.join(full_text)
                    self.add(sentence)
                    sent_rank += 1

                # start a new sentence
                sentence = Sentence()
                full_text = []

            # dpc-bmm-001086-nl-sen   0   Deze   (261
            else:
                fields = line.split()
                sentence.add(Token([
                  '{}'.format(len(sentence) + 1),  # ID = fields[0]
                  fields[2],  # FORM = fields[1]
                  '',  # LEMMA = fields[2]
                  '_',  # UPOS = fields[3]
                  '_',  # XPOS = fields[4]
                  '_',  # FEATS = fields[5]
                  '_',  # HEAD = fields[6]
                  '_',  # DEPREL = fields[7]
                  '_',  # DEPS = fields[8]
                  '_',  # MISC = fields[9]
                  '_',  # FRAME = fields[10]
                  '_',  # ROLE = fields[11]
                  fields[3]  # COREF = fields[12]
                ]))
                full_text.append(fields[2])

    def _load(self, filename):
        logging.info("Opening {}".format(filename))

        with open(filename, "r") as f:
            conllu_raw = f.readlines()

        # sent_id = 116
        # text = Wie kan optreden ?
        # <12 columns tab separated, 1 line per token in the sentence>
        # <empty line>
        sentence = Sentence()
        doc_current_id = os.path.basename(filename)
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
                            filename, len(self.doc_lengths))
            elif line[0:1] == '#':
                # ignore comments
                continue
            elif len(line) == 0:
                # newline means end of a sentence
                if len(sentence) > 0:
                    sentence.doc_id = doc_current_id
                    self.add(sentence)

                # start a new sentence
                sentence = Sentence()
            else:
                fields = line.split('\t')
                sentence.add(Token(fields))

        # if the file does not end in an empty line,
        # assume the sentence is finished and add it anyways
        if len(sentence) > 0:
            sentence.doc_id = doc_current_id
            self.add(sentence)

    def add(self, sentence):
        if sentence.doc_id in self.doc_lengths:
            sentence.sent_rank = self.doc_lengths[sentence.doc_id]
            self.doc_lengths[sentence.doc_id] += 1
        else:
            sentence.sent_rank = 0
            self.doc_lengths[sentence.doc_id] = 1

        sentence.doc_rank = \
            list(self.doc_lengths.keys()).index(sentence.doc_id)

        sentence.dataset = self
        self.sentences.append(sentence)

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


def transform_coordinations(sentence):
    """
    Transform UD coordination to be more like SD coordination.

    Universal Dependency style coordination is not suitable for our head-based
    approach; in the text '.. A and B ..' the possible heads are:

    A which will correspond to the text 'A and B'
    B which will correspond to 'and B'.

    Add an extra node, and transform the dependency graph, to allow selecting:
    'A', 'and B', 'A and B'
    """

    # find all coordinations, ie. tokens with DEPREL = 'conj'
    coordinations = {}
    for token in sentence:
        if token.DEPREL == 'conj':
            # identify the coordinations by the head
            coordination_id = token.HEAD
            if coordination_id in coordinations:
                coordinations[coordination_id].append(token.ID)
            else:
                coordinations[coordination_id] = [token.ID]

    # transform each coordination
    #           ^                           ^
    #           | deprel                    | deprel
    #         tokenA              =>       tokenX
    #       /conj    \ conj          /conj  |conj  \conj
    #   tokenB        tokenC      tokenA   tokenB   tokenC
    for tokenA_id in coordinations:
        coordination = coordinations[tokenA_id]
        tokenA = sentence[tokenA_id]

        # attach tokens directly to the original HEAD of the first token
        for token_id in coordination:
            sentence[token_id].HEAD = tokenA.HEAD

        # create a dummy node to represent the full coordination
        tokenX = Token([
            '{}'.format(len(sentence) + 1),  # ID
            '',  # FORM
            '',  # LEMMA
            '_',  # UPOS
            'mv',  # XPOS
            'Number=Plur',  # FEATS
            tokenA.HEAD,
            tokenA.DEPREL,
            tokenA.DEPS,
            tokenA.MISC,
            tokenA.FRAME,
            tokenA.ROLE,
            '_'  # COREF
            ])

        # attach all tokens to the dummy
        for token_id in coordination:
            sentence[token_id].HEAD = tokenX.ID

        tokenA.HEAD = tokenX.ID
        tokenA.DEPREL = 'conj'
        tokenA.FRAME = '_'
        tokenA.ROLE = '_'

        sentence.add(tokenX)

        # fix remaining issues:
        # - punctuation is a child of the root token
        if tokenX.DEPREL == 'root':
            for token in sentence:
                if token.DEPREL == 'punct':
                    token.HEAD = tokenX.ID

    return sentence


def swap_copula(sentence, noun, verb):
    """
    We promote the copula token to be the head (like other verbs would be),
    and attach the mention to it.  We then try to clean up the graph by moving
    a number of descendants of the old head (ie part the mention) to the
    copula. The choice is based on the DEPREL, for a list see
    COPULA_NOUN_DESC_MOVE_TO_VERB.
    """
    # 1. swap the deprel of the 'cop' and its syntactic head
    # (lets call it its noun)
    verb.DEPREL = noun.DEPREL
    noun.DEPREL = 'cop'
    verb.HEAD = noun.HEAD
    noun.HEAD = verb.ID

    # 2. of the dependents of the noun, move some to the verb
    for token in sentence:
        if token.HEAD == noun.ID:
            if token.DEPREL in COPULA_NOUN_DESC_MOVE_TO_VERB:
                token.HEAD = verb.ID

    return sentence


def transform_copulas(sentence):
    """
    Transform the UD copola usage.

    We focus on Dutch, which has explicit copolas: 'Dat is fantastisch.'
    UD requires 'Dat' to be the head, which leads to mentions spanning the
    whole sentence, which in turn will confuse all our mention and pairwise
    mention features (nesting, matching).

    For cases with multiple linked copulas 'Dat is en blijft fantastisch',
    we set the DEPREL to 'conj', like UD does for normal verbs.
    """
    # 1. gather all 'cop' that point to another 'cop'
    copulas = []
    chained_copulas = []
    for token in sentence:
        if token.DEPREL != 'cop':
            continue
        if token.HEAD == '0':
            # Inconceivable! DEPREL == 'cop', but HEAD == '0'
            return sentence

        if sentence[token.HEAD].DEPREL == 'cop':
            # we have a -cop-> b -cop-> ..
            # turn this into a conjunction
            chained_copulas.append(token)
        else:
            copulas.append(token)

    # if we didnt find any copulas we're done
    if len(copulas) == 0:
        return sentence

    # 2. change the releation to conj
    for token in chained_copulas:
        token.DEPREL = 'conj'

    # 3. swap the noun and verb
    for verb in copulas:
        noun = sentence[verb.HEAD]
        swap_copula(sentence, noun, verb)

    # 4. Clean-up:, all 'punct' must point to the head
    for token in sentence:
        if token.HEAD == '0':
            # found the head
            head_id = token.ID
            break

    for token in sentence:
        if token.DEPREL == 'punct':
            token.HEAD = head_id

    return sentence


def transform_tree(sentence):
    """
    Make dependency graph better suited for our task.
    This calls: transform_copulas and transform_coordinations.
    """
    sentence = transform_copulas(sentence)
    sentence = transform_coordinations(sentence)
    return sentence

def write_output_conll2012(dataset, filename):
    keyfile = open(filename, 'w')

    firstDoc = True
    current_doc = None
    for sentence in dataset:
        if sentence.doc_id != current_doc:
            if firstDoc:
                firstDoc = False
            else:
                keyfile.write('#end document\n')

            current_doc = sentence.doc_id
            keyfile.write('#begin document ({});\n'.format(current_doc))
        else:
            keyfile.write('\n')

        for token in sentence:
            if token.FORM == '':
                # these are from unfolding the coordination clauses, dont print
                if token.COREF != '_':
                    logging.error(
                            'Hidden token has a coref={}'.format(token.COREF)
                            )
                    print(sentence)
                    print()
                continue
            if token.COREF != '_':
                coref = token.COREF
            else:
                coref = '-'
            keyfile.write('{}\t0\t{}\t{}\t{}\n'.format(
                sentence.doc_id, token.ID, token.FORM, coref))

    keyfile.write('#end document\n')
    keyfile.close()

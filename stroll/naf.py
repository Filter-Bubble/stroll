import logging
import sys

from KafNafParserPy import KafNafParser, Clp, Cspan, Cpredicate, Crole
from KafNafParserPy.span_data import Ctarget

from stroll.conllu import ConlluDataset, Sentence, Token


def write_header_to_naf(naf):
    """Write SRL information to the semantic role layer of a NAF file.
    """
    # Include the linguistic processor
    lp = Clp()
    lp.set_name('STROLL-SRL')
    lp.set_version('my_version')  # TODO
    lp.set_timestamp()  # Set to the current date and time
    naf.add_linguistic_processor('srl',lp)


def write_frames_to_naf(naf, frames, sentence):
    for fid in frames:
        # create a span
        # assume a single target, ie len(span) == 1, for the predicate
        span_obj = Cspan()
        span_obj.add_target_id(sentence[frames[fid].ID].nafid)  # a term identifier

        # create a predicate with the span
        pred_obj = Cpredicate()
        pred_obj.set_span(span_obj)
        pred_obj.set_uri('UNSET')  # TODO
        pred_obj.set_confidence('{}'.format(frames[fid].p.numpy()))

        # pred_obj.set_id('pr{}'.format(pred_counter)) # should be like pr\d+
        pred_obj.set_id('pr_s{}t{}'.format(sentence.sent_rank + 1, frames[fid].ID))

        # now add all the roles to the predicate
        for argument in frames[fid].arguments:
            # create a span for this role
            span_obj = Cspan()

            # add targets to the span
            for i in argument['ids']:

                # keep track of the syntactic head
                if i == argument['id']:
                    target_obj = Ctarget()
                    target_obj.set_id(sentence[i].nafid)
                    target_obj.set_head('yes')
                    span_obj.add_target(target_obj)
                else:
                    span_obj.add_target_id(sentence[i].nafid)

            # create the role
            role_obj = Crole()
            role_obj.set_sem_role(argument['role'])
            role_obj.set_id('r_s{}t{}'.format(sentence.sent_rank + 1, argument['id']))  # should be like: r\d+
            role_obj.set_span(span_obj)
            pred_obj.add_role(role_obj)

        # add the predicate to the file
        naf.add_predicate(pred_obj)


def load_naf_stdin():
    """Load a dataset in NAF format.

    Use this function to create a new ConlluDataset from a NAF file,
    read from stdin.

    NOTE: you can only add to NAF files, not create one from scratch.
    """
    my_parser = KafNafParser(sys.stdin)    

    my_dataset = ConlluDataset()

    # a big look-up table: for any NAF id, return a hash with 
    # {sent_id, token_id} in the ConlluDataset
    naf2conll_id = {}

    # collect the sentences in a hash, indexed by token_obj.get_sent()
    sentences = {}

    # iterate over the tokens to get: ID, FORM
    for token_obj in my_parser.get_tokens():
        # (string) identifier of the sentence
        sent_id = token_obj.get_sent()
        if sent_id in sentences:
            sentence = sentences[sent_id]
        else:
            sentence = Sentence(sent_id=sent_id)
            sentences[sent_id] = sentence

        # (string) number of the token in the sentence, starting at '1'
        token_id = '{}'.format(len(sentence) + 1)  # ID

        new_token = Token([
            token_id,  # ID
            token_obj.get_text(),  # FORM
            '_',  # LEMMA
            '_',  # UPOS
            '_',  # XPOS
            '_',  # FEATS
            '0',  # HEAD -> to be overwritten later
            'root',  # DEPREL -> to be overwritten later
            '_',  # DEPS
            '_'   # MISC
            ])

        sentence.add(new_token)

        # to match a NAF span to conll tokens, we need sent_id and token_id
        naf2conll_id[token_obj.get_id()] = {
            'sent_id': sent_id,
            'token_id': token_id
            }

    # iterate over the term to get: LEMMA, XPOS, UPOS, FEATS, sent_id, nafid
    for term_obj in my_parser.get_terms():
        # span
        # TODO: for now, assume terms map one-on-one on tokens
        nafid = term_obj.get_span().get_span_ids()
        if len(nafid) > 1:
            logging.error('Multi-word tokens not implemented yet.')
            return
        nafid = nafid[0]

        conllid = naf2conll_id[nafid]
        sent_id = conllid['sent_id']
        sentence = sentences[sent_id]

        token_id = conllid['token_id']
        token = sentence[token_id]

        # store the identifier of the NAF term on the token, so we can add
        # information to the NAF later.
        token.nafid = term_obj.get_id()

        token.LEMMA = term_obj.get_lemma()

        # NAF pos='' is in lower case, UD UPOS is upper case
        token.UPOS = term_obj.get_pos().upper()

        # naf: A(B,C) -> ud: A|B|C
        xpos = term_obj.get_morphofeat()
        if xpos:
            token.XPOS = xpos.replace('(', '|').replace(')', '').replace(',','|')
            if token.XPOS[-1] == '|':
                token.XPOS = token.XPOS[:-1]

        # look for an external reference containing FEATS
        for ext_ref in term_obj.get_external_references():
            if ext_ref.get_reftype() == 'FEATS':
                token.FEATS = ext_ref.get_reference()

        # to match NAF dependencies to conll tokens, we need sent_id and token_id
        naf2conll_id[term_obj.get_id()] = {
            'sent_id': sent_id,
            'token_id': token_id
            }

    # iterate over the dependencies to get: HEAD, DEPREL
    for dep_obj in my_parser.get_dependencies():
        # from
        conllid = naf2conll_id[dep_obj.get_from()]
        sent_id = conllid['sent_id']
        sentence = sentences[sent_id]

        token_id = conllid['token_id']
        token_from = sentence[token_id]

        # to
        conllid = naf2conll_id[dep_obj.get_to()]
        sent_id = conllid['sent_id']
        sentence = sentences[sent_id]

        token_id = conllid['token_id']
        token_to = sentence[token_id]

        # function
        depfunc = dep_obj.get_function()

        token_to.HEAD = token_from.ID
        token_to.DEPREL = depfunc

    # A final conversion of our list of sentences to a ConlluDataset
    for sent_id in sentences:
        sentence = sentences[sent_id]

        # construct the sentence.full_text
        raw_tokens = []
        for token in sentence:
            raw_tokens.append(token.FORM)
        sentence.full_text = ' '.join(raw_tokens)

        # add to the dataset
        my_dataset.add(sentence)

    my_dataset.naf2conll_id = naf2conll_id

    return my_dataset, my_parser

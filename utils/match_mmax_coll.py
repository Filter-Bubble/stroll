import sys

import re
from stroll.conllu import ConlluDataset, Sentence, Token

# #begin document (WR-P-E-C-0000000004); part 000
docid_start_pattern = re.compile('#begin document \((.*)\); part \d+$')
#end document
docid_end_pattern = re.compile('#end document')
emptyline_pattern = re.compile('^\w*$')
wikidocid_pattern = re.compile('wiki(\d+)')

too_short_docs = 0
too_long_docs = 0
too_short_words = 0
too_long_words = 0
complete_docs = 0
matched_lines = 0
matched_words = 0
missed_words = 0


def fix_docid(docid):
    match = wikidocid_pattern.match(docid)
    if match:
        number, = match.groups(0)
        docid = 'wiki-{}'.format(number)
    return docid


def find_sentences(docid, dataset):
    matching_sentences = []
    docidmatch = re.compile('^.*' + docid + '.*\.p\.\d+\.s\.\d+.*$')
    docidmatch_alt = re.compile('^.*' + docid + '.*\d+.*$')

    for sentence in dataset:
        match = docidmatch.match(sentence.sent_id)
        alt_match = docidmatch_alt.match(sentence.sent_id)
        if match:
            matching_sentences.append(sentence)
        elif alt_match:
            matching_sentences.append(sentence)

    return matching_sentences


def do_match(docid, doc, parsed):
    print('Document {} contains {} sentences, {} candidates.'.format(
        docid, len(doc), len(parsed)
        ))

    # build a hash from the candidates text to candidate id
    candidates = {}
    for i, sentence in enumerate(parsed):
        full_text = sentence.full_text.strip().replace(' ', '')
        candidates[full_text] = i

    failure = False
    result = []

    for doc_sentence in doc:
        full_text = doc_sentence.full_text.strip().replace(' ', '')
        if full_text in candidates:
            match = parsed[candidates[full_text]]

            i=0
            j=0
            while i < len(match): # loop over each token in the match
                # take the next token from the match
                match_form = match[i].FORM
                i += 1

                # take the next token and coref annotation from the doc
                doc_form = doc_sentence[j].FORM
                coref = doc_sentence[j].COREF
                j += 1

                # is the doc token larger?
                while len(match_form) < len(doc_form):
                    # keep adding match tokens until we match
                    match_form += match[i].FORM
                    i += 1

                # is the match token larger?
                while len(match_form) > len(doc_form):
                    # keep adding doc tokens until we match
                    doc_form += doc_sentence[j].FORM
                    if doc_sentence[j].COREF != '_':
                        if coref == '_':
                            coref = doc_sentence[j].COREF
                        else:
                            coref += '|' + doc_sentence[j].COREF
                    j += 1

                # add coref
                match[i-1].COREF = coref

            result.append(match)

    if failure:
        return None
    else:
        return result


with open('mmax.conll', 'r') as f:
    mmax = f.readlines()

dataset = ConlluDataset('lassy.conllu')

docid = None
doc_sentences = None
current_sentence = None

for line in mmax:
    line = line.strip()

    start_match = docid_start_pattern.match(line)
    end_match = docid_end_pattern.match(line)
    empty_match = emptyline_pattern.match(line)

    if empty_match:
        # finalize sentence
        current_sentence.full_text = ' '.join([token.FORM for token in current_sentence]) 
        doc_sentences.append(current_sentence)

        # start a new sentence
        current_sentence = Sentence(sent_id='{}.{}'.format(docid, len(doc_sentences)))
    elif line[0] == '#':
        if start_match:
            # start with a new doc
            docid, = start_match.groups(0)
            docid = fix_docid(docid)
            doc_sentences = []

            # start a new sentence
            current_sentence = Sentence(sent_id='{}.{}'.format(docid, len(doc_sentences)))
        elif end_match:
            # Don't finalize sentence, as the #end is preceded by a newline

            # finalize doc
            parsed_sentences = find_sentences(docid, dataset)
            results = do_match(docid, doc_sentences, parsed_sentences)
            if results:
                with open('docs/{}.conll'.format(docid), 'w') as out:
                    for result in results:
                        out.write(result.__repr__())
                        out.write('\n\n')
            else:
                print('{}: missing'.format(docid))
                pass

            doc_sentences = None
        else:
            pass
    else:
        fields = line.split('\t')
        new_token = Token([
            fields[1],  # ID
            fields[2],  # FORM
            '_',  # LEMMA
            '_',  # UPOS
            '_',  # XPOS
            '_',  # FEATS
            '_',  # HEAD
            '_',  # DEPREL
            '_',  # DEPS
            '_',  # MISC
            ])
        new_token.COREF = fields[3]
        current_sentence.add(new_token)

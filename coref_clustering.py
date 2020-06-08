import argparse
import logging

from stroll.graph import ConlluDataset
from stroll.coref import preprocess_sentence, postprocess_sentence

from stroll.coref import mentions_match_exactly
from stroll.coref import mentions_match_relaxed

parser = argparse.ArgumentParser(
        description='Create coreference chains from a similarity matrix',
        )
parser.add_argument(
        '--conll',
        help='Annotated conll dataset'
        )


def build_scoring_file(dataset, docname, filename):
    keyfile = open(filename, 'w')
    keyfile.write('#begin document ({});\n'.format(args.conll))
    isFirst = True
    for sentence in dataset:
        if isFirst:
            isFirst = False
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


def external_scoring(args):
    # 1. load conll file
    input_set = ConlluDataset(args.conll)

    # 2. build key file to score against
    build_scoring_file(input_set, args.conll, 'clustering.key')

    # 3. pre-process the dependency tree to unfold coordination
    #   and convert the gold span based mentions to head-based mentions
    for sentence in input_set:
        preprocess_sentence(sentence)

    # 4. this is where we should do our coref resolution

    # 5. post-process the sentence to get spans again
    for sentence in input_set:
        postprocess_sentence(sentence)

    # 6. write out our sentence for scoring
    build_scoring_file(input_set, args.conll, 'clustering.response')


def internal_scoring(args):
    # 1. load conll file
    input_set = ConlluDataset(args.conll)

    logging.info('Processing')
    targets = []
    candidates = []
    for sentence in input_set:
        # 2. pre-process the dependency tree to unfold coordination
        #    keep track of both target and candidates
        #    m_braket: gold mentions from the bra-ket annotations
        #    m_head: gold mentions after mapping bra-ket to heads
        m_braket, m_head = preprocess_sentence(sentence)
        targets += m_braket
        candidates += m_head

        # 3. run network
        # 4. get the system mentions

    # 5. score
    logging.info('Scoring')
    print('Gold:         ', len(targets))
    print('System:       ', len(candidates))

    # First find exact matches
    exact_match = 0
    logging.info('Scoring exact')

    unmatched_targets = []
    while len(targets):
        target_matched = False
        target = targets.pop()
        for candidate in candidates:
            if target.sentence != candidate.sentence:
                continue
            if mentions_match_exactly(target, candidate):
                exact_match += 1
                target_matched = True
                candidates.remove(candidate)
                break
        if not target_matched:
            unmatched_targets.append(target)

    # Then find relaxed matches
    relaxed_match = 0
    logging.info('Scoring relaxed')

    targets = unmatched_targets
    unmatched_targets = []
    while len(targets):
        target_matched = False
        target = targets.pop()
        for candidate in candidates:
            if target.sentence != candidate.sentence:
                continue

            if mentions_match_relaxed(target, candidate):
                relaxed_match += 1
                target_matched = True
                candidates.remove(candidate)
                break
        if not target_matched:
            unmatched_targets.append(target)

    # Report
    print('Exact match:  ', exact_match)
    print('Relaxed match:', relaxed_match)
    print('Missed:       ', len(unmatched_targets))
    print('Invented:     ', len(candidates))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    # internal_scoring(args)
    dataset = ConlluDataset(args.conll)

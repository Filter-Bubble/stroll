#!/usr/bin/env python3
import argparse

from stroll.conllu import ConlluDataset, write_output_conll2012
from stroll.coref import preprocess_sentence, postprocess_sentence

import logging

parser = argparse.ArgumentParser(
        description='Convert CONLL-U file with coreferences in head \
                notation to CONLL2012 files with span based coreferences'
        )
parser.add_argument(
        'input',
        help='Input file in CoNLL-U or CoNLL2012 format'
        )

parser.add_argument(
        'output',
        help='Output file in CONLL2012 format',
        )
parser.add_argument(
        '-i', '--input_format',
        choices=['conllu', 'conll2012'],
        help='Format of input file',
        default='conllu'
        )
parser.add_argument(
        '-o', '--output_format',
        choices=['conllu', 'conll2012'],
        help='Format of output file',
        default='conll2012'
        )
parser.add_argument(
        '--preprocess',
        help='Apply preprocessing step (only necessary for gold files)',
        action='store_true'
        )

parser.add_argument(
        '--postprocess',
        help='Apply postprocessing step (head mentions to mention spans)',
        action='store_true'
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    if args.input_format == 'conllu':
        dataset = ConlluDataset(args.input)
    else:
        dataset = ConlluDataset()
        dataset.load_conll2012(args.input)

    if args.preprocess:
        for sentence in dataset:
            _, mentions = preprocess_sentence(sentence)

    if args.postprocess:
        for sentence in dataset:
            postprocess_sentence(sentence)

    if args.output_format == 'conll2012':
        write_output_conll2012(dataset, args.output)
    else:
        with open(args.output, 'w') as outfile:
            outfile.write(dataset.__repr__())

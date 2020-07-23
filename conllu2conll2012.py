#!/usr/bin/env python3
import argparse
import re

from stroll.conllu import ConlluDataset, Sentence, Token, write_output_conll2012
from stroll.coref import preprocess_sentence, postprocess_sentence

import logging

parser = argparse.ArgumentParser(
        description='Convert CONLL-U file with coreferences in head notation to CONLL2012 files with span based coreferences'
        )
parser.add_argument(
        'input',
        help='Input file in CoNLLU format'
        )

parser.add_argument(
        'output',
        help='Output file in CONLL2012 format',
        )
parser.add_argument(
        '--preprocess',
        help='Apply preprocessing step (only necessary for gold files)',
        default=False,
        action='store_true'
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    dataset = ConlluDataset(args.input)

    if args.preprocess:
        for sentence in dataset:
            _, mentions = preprocess_sentence(sentence)
            
    for sentence in dataset:
        postprocess_sentence(sentence)

    write_output_conll2012(dataset, args.output)

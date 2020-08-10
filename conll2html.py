#!/usr/bin/env python3
import argparse

from stroll.conllu import ConlluDataset
from jinja2 import FileSystemLoader, Environment


parser = argparse.ArgumentParser(
        description='Run the Stanza parser and produce CoNLL output',
        )
parser.add_argument(
        '--output',
        help='Output filename'
        )
parser.add_argument('-t', '--type', choices=['conll', 'conll2012'], default='conll')
parser.add_argument(
        'input',
        help='Input files'
        )


def write_html(dataset, name):
    loader = FileSystemLoader('.')
    env = Environment(loader=loader)
    template = env.get_template('highlighted.template')

    documents = {}
    entities = {}
    for sentence in dataset:
        doc_id = sentence.doc_id
        if doc_id in documents:
            documents[doc_id]['sentences'].append(sentence)
        else:
            documents[doc_id] = {
                    'doc_id': doc_id,
                    'sentences': [sentence]
                    }
        for token in sentence:
            if token.COREF != '_':
                for ref in token.COREF.split('|'):
                    entities[ref.replace('(', '').replace(')', '')] = 1

    with open(name, 'w') as f:
        f.write(template.render(
            documents=list(documents.values()),
            entities=list(entities.keys())
            )
        )


if __name__ == '__main__':
    args = parser.parse_args()
    if args.type == 'conll2012':
        dataset = ConlluDataset()
        dataset.load_conll2012(args.input)
    else:
        dataset = ConlluDataset(args.input)
    write_html(dataset, args.output)

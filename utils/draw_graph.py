import argparse

from stroll.graph import ConlluDataset

import pygraphviz as pgv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
        description='Train a R-GCN for Semantic Roll Labelling.'
        )
parser.add_argument(
        '--conllu',
        dest='conllu',
        required='true',
        help='Conllu file containing the sentences to draw'
        )
parser.add_argument(
        'sent_ids',
        nargs='+',
        default='*',
        help='Sentence ids to draw'
        )


def draw_graph(sentence, filename):
    G = pgv.AGraph(strict=False, directed=True)

    # add nodes
    for token in sentence:
        if token.FRAME == 'rel':
            if token.ROLE != '_':
                G.add_node(
                        'n' + token.ID,
                        label=token.FORM + '\n' + token.ROLE,
                        shape='box', fillcolor='red', style='filled')
            else:
                G.add_node(
                        'n' + token.ID, label=token.FORM,
                        shape='box', fillcolor='red', style='filled')
        elif token.ROLE != '_':
            G.add_node(
                    'n' + token.ID, label=token.FORM + '\n' + token.ROLE,
                    shape='box', fillcolor='blue', fontcolor='white',
                    style='filled')
        else:
            G.add_node('n' + token.ID, label=token.FORM,
                       shape='box')
    for token in sentence:
        if token.HEAD != '0':
            G.add_edge('n' + token.ID,  'n' + token.HEAD,
                       label=token.DEPREL)

    # default to neato
    G.layout()

    # write previously positioned graph to PNG file
    G.draw(filename)

    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()

    conllu_set = ConlluDataset(args.conllu)

    for sentence in conllu_set:
        if args.sent_ids != '*' and sentence.sent_id not in args.sent_ids:
            continue
        print('Drawing', sentence)
        draw_graph(sentence, sentence.sent_id + '.png')

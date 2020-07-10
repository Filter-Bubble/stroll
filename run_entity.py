import argparse
import logging

import torch
import fasttext

from jinja2 import FileSystemLoader, Environment

from scorch.scores import muc, b_cubed, ceaf_e

from stroll.conllu import ConlluDataset
from stroll.coref import preprocess_sentence, postprocess_sentence
from stroll.coref import get_mentions

from stroll.model import EntityNet
from stroll.entity import Entity
from stroll.entity import action_new_probability, action_add_probabilities
from stroll.entity import set_wordvector

MAX_CANDIDATES = 50

parser = argparse.ArgumentParser(
        description='Run an entity centric trainsition based  coreference net'
        )
parser.add_argument(
        'input',
        help='Input file in CoNLL format'
        )
parser.add_argument(
        '--model',
        default='models/entity.pt',
        help='Trained EntityNet to use',
        )
parser.add_argument(
        '--mmax',
        help='Output file in MMAX format',
        )
parser.add_argument(
        '--html',
        help='Output file in html format',
        )
parser.add_argument(
        '--score',
        help='Score using gold annotation from the input',
        default=False,
        action='store_true'
        )
parser.add_argument(
        '--verbose',
        help='Print candidate entities and probiblities during run',
        default=False,
        action='store_true'
        )
parser.add_argument(
        '--preprocess',
        help='Apply preprocessing step (only necessary for gold files)',
        default=False,
        action='store_true'
        )
parser.add_argument(
        '--output',
        help='Output file in conllu format',
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


def write_output_mmax(dataset, filename):
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


def eval(net, doc):
    net.eval()

    # start without entities
    entities = []

    # add the mentions one-by-one to the entities
    for mention in doc:
        # short cut for single action
        if len(entities) == 0:
            new_entity = Entity()
            new_entity.refid = len(entities)
            new_entity.add(mention)
            entities.append(new_entity)
            continue

        # sort entities by distance
        ranking = []
        for entity in entities:
            top = entity.mentions[-1]
            rank = top.sentence.sent_rank
            ranking.append([entity, rank])
        ranking.sort(key=lambda k: -k[1])

        # take top N entities
        if len(ranking) > MAX_CANDIDATES:
            ranking = ranking[0:MAX_CANDIDATES]
        candidates = [rank[0] for rank in ranking]

        # score the most likely action as predicted by our network
        # MAX_CANDIDATES+1 -> MAX_CANDIDATES+1
        all_probs = torch.cat([
            action_new_probability(net, entities, mention),
            action_add_probabilities(net, candidates, mention)
            ])
        action = all_probs.argmax().item()

        if args.verbose:
            print('----------==', mention.sentence[mention.head].FORM,
                  'mwt: ', ', '.join(list(mention.mwt)))
            print('0 New', all_probs[0].item())
            for c in range(len(candidates)):
                cand = candidates[c]
                print(
                        c+1, 'Add',
                        cand.refid,
                        'Nouns: ' + ' '.join(list(cand.nouns)),
                        'Proper Nouns: ' + ' '.join(list(cand.proper_nouns)),
                        'Pronouns: ' + ' '.join(list(cand.pronouns)),
                        all_probs[c+1].item()
                        )
            print('-.-.-.-.-.-.-', action)

        if action == 0:
            # start a new entity
            new_entity = Entity()
            new_entity.add(mention)
            new_entity.refid = len(entities)
            entities.append(new_entity)
        else:
            # add to existing entity
            existing_entity = candidates[action - 1]
            existing_entity.add(mention)

    # score the entities
    # build list of sets for both gold and system
    gold_sets = {}
    for mention in doc:
        if mention.refid not in gold_sets:
            gold_sets[mention.refid] = set()
        gold_sets[mention.refid].add(mention.get_identifier())
    gold_sets = list(gold_sets.values())

    system_sets = []
    for entity in entities:
        system_sets.append(entity.as_set())

    score_muc = muc(gold_sets, system_sets)
    score_b3 = b_cubed(gold_sets, system_sets)
    score_ce = ceaf_e(gold_sets, system_sets)

    # write results back to entities and dataset
    for refid, entity in enumerate(entities):
        if args.verbose:
            print(entity)
        for mention in entity.mentions:
            if args.verbose:
                print(mention)
            mention.refid = refid
            mention.sentence[mention.head].COREF = '{}'.format(refid)
        if args.verbose:
            print('= - = - = - = - = - = - =')

    return score_muc, score_b3, score_ce


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    # 1. load the MentionNet configuration
    state_dict = torch.load(args.model)
    hyperparams = state_dict.pop('hyperparams')

    # Load the FastText model
    ft = '/home/jiska/Code/ernie/resources/fasttext.model.bin'
    logging.info('Loading default wordvectors {}'.format(ft))
    set_wordvector(fasttext.load_model(ft))

    # 2. load conll file
    dataset = ConlluDataset(args.input)

    # 3. pre-process the dependency tree to unfold coordination
    #    and group the mentions per document
    eval_mentions = []

    if args.preprocess:
        for sentence in dataset:
            _, mentions = preprocess_sentence(sentence)
            eval_mentions += mentions
    else:
        for sentence in dataset:
            eval_mentions += get_mentions(sentence)

    if args.score:
        print('Number of mentions: {}'.format(len(eval_mentions)))

    eval_docs = [[] for i in dataset.doc_lengths]
    for mention in eval_mentions:
        sentence = mention.sentence
        doc_rank = sentence.doc_rank
        eval_docs[doc_rank].append(mention)

    # 5. initialize the network
    net = EntityNet()
    net.load_state_dict(state_dict, strict=False)

    # 6. score mentions
    for doc in eval_docs:
        score_muc, score_b3, score_ce = eval(net, doc)
        if args.score:
            print('\nMuc: {}\n B3:  {}\n Ce:  {}\nCnl: {}'.format(
                score_muc, score_b3, score_ce, [
                    (score_muc[0] + score_b3[0] + score_ce[0]) / 3.,
                    (score_muc[1] + score_b3[1] + score_ce[1]) / 3.,
                    (score_muc[2] + score_b3[2] + score_ce[2]) / 3.
                    ]))

    if args.output:
        with open(args.output, 'w') as f:
            f.write(dataset.__repr__())

    # 3. convert head-based mentions to span-based mentions
    for sentence in dataset:
        postprocess_sentence(sentence)

    if args.mmax:
        write_output_mmax(dataset, args.output)
    if args.html:
        write_html(dataset, args.html)

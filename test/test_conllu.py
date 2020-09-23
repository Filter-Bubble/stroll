import stroll.conllu
import os

__here__ = os.path.dirname(os.path.realpath(__file__))


def test_empty_conll():
    dataset = stroll.conllu.ConlluDataset()
    assert len(dataset.sentences) == 0


def test_load_conllu():
    input_file = os.path.join(__here__, 'data', 'test.conllu')
    dataset = stroll.conllu.ConlluDataset(input_file)
    assert len(dataset.sentences) == 1
    sent = dataset[0]
    assert len(sent) == 11
    tok = sent[0]
    assert tok.COREF == '(0)'
    assert tok.COREF_HEAD == None

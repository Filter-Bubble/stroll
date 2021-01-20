import stanza
import torch
import dgl

from pathlib import Path

from stroll.conllu import Token, Sentence, ConlluDataset
from stroll.graph import GraphDataset
from stroll.labels import FasttextEncoder, get_dims_for_features
from stroll.model import Net
from stroll.download import download_srl_model
from stroll.srl import predict

from stanza.pipeline.processor import Processor, register_processor
from stanza.models.common.doc import Document

from torch.utils.data import DataLoader

def srlSetter(self, value):
    self._srl = value

def frameSetter(self, value):
    self._frame = value

stanza.models.common.doc.Word.add_property('srl', default='_', setter=srlSetter)
stanza.models.common.doc.Word.add_property('frame', default='_', setter=frameSetter)


@register_processor('srl')
class SrlProcessor(Processor):
    ''' Processor that appends semantic roles '''
    _requires = set(['tokenize', 'pos', 'lemma', 'depparse'])
    _provides = set(['srl'])
    
    def __init__(self, config, pipeline, use_gpu):
        # get Paths to default SRL and FastText models
        datapath = Path(config['model_path']).parent
        fname_fasttext, fname_model = download_srl_model(datapath=datapath)

        state_dict = torch.load(fname_model)

        hyperparams = state_dict.pop('hyperparams')

        self.features = hyperparams.features

        in_feats = get_dims_for_features(hyperparams.features)
        if 'WVEC' in hyperparams.features:
            self.sentence_encoder = FasttextEncoder(fname_fasttext)
            in_feats += self.sentence_encoder.dims
        else:
            self.sentence_encoder = None

        self.net = Net(
            in_feats=in_feats,
            h_layers=hyperparams.h_layers,
            h_dims=hyperparams.h_dims,
            out_feats_a=2,
            out_feats_b=19,
            activation='relu'
        )
        self.net.load_state_dict(state_dict)

    def _set_up_model(self, *args):
        print ('_set_up_model')
        pass

    def process(self, doc):
        # convert Document to ConlluDataset
        dicts = doc.to_dict()
        dataset = ConlluDataset()

        for sent_id, input_sentence in enumerate(doc.sentences):
            sentence = Sentence()
            for w in input_sentence.words:
                feats = w.feats if w.feats else '_'
                token = Token([
                  str(w.id),  # ID
                  w.text,  # FORM
                  w.lemma,  # LEMMA
                  w.upos,  # UPOS
                  w.xpos,  # XPOS
                  feats,  # FEATS
                  str(w.head),  # HEAD
                  w.deprel,  # DEPREL
                  '_',  # DEPS
                  '_'  # MISC
                ])
                sentence.add(token)

            sentence.full_text = ''
            sentence.doc_id = ''
            sentence.sent_id = str(sent_id)
            dataset.add(sentence)

        # run stroll
        eval_set = GraphDataset(
            dataset=dataset,
            sentence_encoder=self.sentence_encoder,
            features=self.features
        )
        evalloader = DataLoader(
            eval_set,
            batch_size=50,  # TODO make configurable
            num_workers=2,  # TODO make configurable
            collate_fn=dgl.batch
        )

        self.net.eval()
        with torch.no_grad():
            sent_id = 0
            for gs in evalloader:

                frame_labels, role_labels, \
                    frame_chance, role_chance = self.net.label(gs)

                word_offset = 0
                for g in dgl.unbatch(gs):
                    input_sentence = doc.sentences[sent_id]
                    for w, word in enumerate(input_sentence.words):
                        word.srl = role_labels[w + word_offset]
                        word.frame = frame_labels[w + word_offset]
                    word_offset += len(g)
                    sent_id += 1

        return doc

#!/usr/bin/env bash
filename=$1
doc="${filename%.*}"
echo "Processing file: $filename"


echo "Parsing with stanza..."
python run_stanza.py --nogpu --keep_coref -f conllu --output ${doc}_stanza.conll ${filename}

echo "Preprocessing stanza file.."
python conll2conll.py --preprocess -i conllu -o conllu ${doc}_stanza.conll ${doc}_stanza_prep.conll

# COREF: two options:

echo "Running coref with mention similarity..."
# run stroll for coreference resolution using the mention-mention similarity ala neuralcoref
python3 run_coref.py --output ${doc}_goldmentions_coref.conll --conll2012 ${doc}_goldmentions_coref.2012.conll ${doc}_stanza_prep.conll

echo "Running entity matching..."
# run stroll for coreference resolution using entity matching
python3 run_entity.py --conll2012 ${doc}_goldmentions_entity.2012.conll --output ${doc}_goldmentions_entity.conll ${doc}_stanza_prep.conll

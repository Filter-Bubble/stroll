#!/usr/bin/env bash
filename=$1
doc="${filename%.*}"
echo "Processing file: $filename"

## run the stanford parser to get POS, LEMMA, and DEP
# Assume input is conll file
python run_stanza.py --nogpu -f conllu --output ${doc}_stanza.conll ${filename}

# run stroll for semantic role labelling
python postprocess_srl.py --output ${doc}_srl.conll ${doc}_stanza.conll

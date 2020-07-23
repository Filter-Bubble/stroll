#!/usr/bin/env bash
filename=$1
doc="${filename%.*}"
prepfile=${doc}_preprocessed.conll
echo "Processing file: $filename"

echo "Preprocessing gold file.."
python conll2conll.py --preprocess -i conllu -o conllu $filename $prepfile

# COREF: two options:

echo "Running coref with mention similarity..."
# run stroll for coreference resolution using the mention-mention similarity ala neuralcoref
python3 run_coref.py --output ${doc}_goldparsementions_coref.conll --conll2012 ${doc}_goldparsementions_coref.2012.conll $prepfile

echo "Running entity matching..."
# run stroll for coreference resolution using entity matching
python3 run_entity.py --conll2012 ${doc}_goldparsementions_entity.2012.conll --output ${doc}_goldparsementions_entity.conll $prepfile

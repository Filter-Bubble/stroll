#!/usr/bin/env bash
filename=$1
doc="${filename%.*}"
echo "Processing file: $filename"


# run stroll for semantic role labelling
python postprocess_srl.py --output ${doc}_srl.conll ${filename}

# run stroll for mention detection
# this also transforms the dependency graph, to be a bit more like stanford dependencies, instead of universal deps.
python3 run_mentions.py --output ${doc}_mentions.conll ${doc}_srl.conll

# COREF: two options:
# run stroll for coreference resolution using the mention-mention similarity ala neuralcoref
python3 run_coref.py --html ${doc}_coref.html --output ${doc}_coref.conll --conll2012 ${doc}_coref.2012.conll ${doc}_mentions.conll

# run stroll for coreference resolution using entity matching
python3 run_entity.py --html ${doc}_entity.html --conll2012 ${doc}_entity.2012.conll --output ${doc}_entity.2012.conll ${doc}_mentions.conll

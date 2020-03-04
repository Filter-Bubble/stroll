# Graph based Semantic Role Labeller

This is a semantic role laballer based on a graph convolutional network.
The goal is to make something reasonably state-of-the-art for the Dutch language.

This is work in progress.

## Training data

Sonar contains a annotated dataset of 500K words.
The annotation of the frames is quite basic:
 * the semantic annotations were made on top of a syntactic dependency tree
 * only (some) verbs are marked
 * no linking to FrameNet or any frame identification is made

The original annotations were made on AlpinoXML; the files are converted to conllu using a script from Groningen.

I am planning to use universal dependencies schema for the syntax, and assume input files are in conllu format.
This way the labeller can be used as part of the Newsreader annotation pipeline.

# Installation

Setup a virtual env with python3, and install dependencies:
```bash
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```

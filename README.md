# Graph based Semantic Role Labeller

This is a semantic role laballer based on a graph convolutional network.
The goal is to make something reasonably state-of-the-art for the Dutch language.

This is work in progress.

# Quick start

## Run it on Conll files

```
python -m stroll.srl --dataset example.conll
```

## Use it in a Stanza Pipeline directly from python

You can add Stroll to a Stanza pipeline by importing ```stroll.stanza``` and
adding ```srl``` to the Stanza processors.
This will add an *srl* and *frame* attribute to words.
(Note that these are not printed when printing the Stanza Document, Sentence, or Word objects.)

```
import stanza
import stroll.stanza

nlp = stanza.Pipeline(lang='nl', processors='tokenize,lemma,pos,depparse,srl')
doc = nlp('Stroll annoteert semantic roles.')

for s in doc.sentences:
    for w in s.words:
        print(w.lemma, w.srl, w.frame)

Stroll Arg0 _
annoteren _ rel
semantic Arg1 _
roles _ _
. _ _
```

## Training data

[SoNaR](http://lands.let.ru.nl/projects/SoNaR/) contains a annotated dataset of 500K words.
The annotation of the frames is quite basic:
 * the semantic annotations were made on top of a syntactic dependency tree
 * only (some) verbs are marked
 * no linking to FrameNet or any frame identification is made

The original annotations were made on AlpinoXML; the files are converted to conllu using a script from Groningen.
Conversion scripts from Alpino and Lassy to the UD format [available here](https://github.com/gossebouma/lassy2ud).

I am planning to use universal dependencies schema for the syntax, and assume input files are in conllu format.
This way the labeller can be used as part of the Newsreader annotation pipeline.

## Approach

The training set was made as annotations on top of a syntactic tree.
We'll use the same tree, as given by the `HEAD` and `DEPREL` fields from the conll files.
Words form the nodes, and we add three kinds of edges:
 * from dependent to head (weighted by the number of dependents)
 * from the head to the dependent
 * from the node to itself

At each node, we add a `GRU` cell.
The initial state is made using one-hot encoding of a number of features.
The output of the cell is then passed to two classifiers
One to indicate if this word is a frame, and one to indicate if the word is the head of an arguments (and which argument).

As node features we can use the information in the conllu file.
We also added pre-trained word vectors form either fasttext, or BERTje.
Finally, we add a positional encoding (ie. 'first descendant').

# Installation

Setup a virtual env with python3, and install dependencies:
```bash
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```

# Best model until now

## Model layers

```
Net(
  (embedding): Sequential(
    (0): Linear(in_features=189, out_features=100, bias=True)
    (1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (gru): RGRUGCN(
    in_feats=100, out_feats=100
    (gru): GRU(100, 100)
    (batchnorm): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (task_a): MLP(
    (fc): Sequential(
      (0): Linear(in_features=100, out_features=100, bias=True)
      (1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=100, out_features=2, bias=True)
    )
  )
  (task_b): MLP(
    (fc): Sequential(
      (0): Linear(in_features=100, out_features=100, bias=True)
      (1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=100, out_features=21, bias=True)
    )
  )
)
```

## Training settings

 * Features used UPOS, DEPREL, FEATS, RID,  FastText100
 * Size of hidden state 100
 * Number of GRU iterations 4
 * Activation functions relu
 * Batch size 50 sentences
 * Adam optimizer ADAM
 * Initial learning rate 1e-02
 * Learning rate scheduler StepLR, gamma=0.9
 * Focal Loss, gamma=5.
 * The two loss functions were added, both with weight 1

## Results

2 classes are so rare, they are not in our 10% evaluation set, and were not predicted by the model. (`Arg5` and `ArgM-STR`).

The confustion matrix and statistics (`classification_report`) were made with `scikit-learn`.

The best model was after 6628920 words, or 15 epochs.

### Frames

|   |   _   |  rel |
|---|------:|-----:|
| _ | 45602 |  152 |
|rel|  164  | 3561 |


|             | precision   | recall  |f1-score   |support|
|-------------|------------:|--------:|----------:|------:|
|           _ |      1.00   |   1.00  |    1.00   |  45754|
|         rel |      0.96   |   0.96  |    0.96   |   3725|
|             |             |         |           |       |
|    accuracy |             |         |    0.99   |  49479|
|   macro avg |      0.98   |   0.98  |    0.98   |  49479|
|weighted avg |      0.99   |   0.99  |    0.99   |  49479|


### Roles

|         | Arg0|   Arg1|  Arg2| Arg3 | Arg4  |ArgM-ADV | ArgM-CAU  | ArgM-DIR | ArgM-DIS | ArgM-EXT | ArgM-LOC | ArgM-MNR | ArgM-MOD | ArgM-NEG | ArgM-PNC | ArgM-PRD | ArgM-REC | ArgM-TMP  |  _   |
|---------|----:|------:|-----:|-----:|------:|--------:|----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|-----:|
|     Arg0| 1650|   113 |    9 |    0 |    0  |       2 |         2 |        1 |       0  |       0  |        3 |         0|       0  |        0 |        2 |        0 |        0 |          0|    32|
|     Arg1|  232|  2596 |  111 |    0 |    6  |       4 |         3 |        3 |       0  |       5  |       41 |        17|       1  |        6 |        7 |        2 |        2 |          2|   100|
|     Arg2|   16|   154 |  346 |    2 |    8  |       1 |         0 |       10 |       0  |       3  |       91 |        27|       0  |        1 |       16 |        6 |        2 |          6|    18|
|     Arg3|    0|     9 |   26 |    5 |    0  |       0 |         0 |        1 |       0  |       0  |        7 |         3|       0  |        0 |        8 |        1 |        0 |          4|     1|
|     Arg4|    0|     5 |   11 |    0 |   28  |       0 |         0 |        3 |       0  |       0  |        7 |         0|       0  |        0 |        0 |        0 |        0 |          1|     0|
| ArgM-ADV|    0|     8 |   13 |    0 |    0  |     312 |        10 |        2 |      27  |       5  |       20 |        32|       0  |        0 |        4 |        6 |        0 |         45|    44|
| ArgM-CAU|    9|     5 |    5 |    0 |    0  |      13 |        96 |        1 |       1  |       0  |        0 |        20|       0  |        0 |        1 |        0 |        0 |          7|    12|
| ArgM-DIR|    0|     0 |   10 |    0 |    4  |       0 |         0 |       30 |       0  |       0  |        4 |         5|       0  |        0 |        0 |        1 |        0 |          2|     2|
| ArgM-DIS|    0|     0 |    0 |    0 |    0  |      27 |         3 |        0 |     322  |       2  |        5 |        13|       0  |        0 |        0 |        0 |        0 |         14|   157|
| ArgM-EXT|    0|     4 |    4 |    1 |    0  |       6 |         0 |        0 |       0  |      47  |        2 |         8|       0  |        1 |        0 |        0 |        0 |          9|     5|
| ArgM-LOC|    0|    12 |   49 |    0 |    3  |       8 |         1 |        7 |       0  |       1  |      519 |         6|       0  |        1 |        4 |        2 |        1 |         26|    36|
| ArgM-MNR|    3|    25 |   35 |    0 |    0  |      38 |         6 |        8 |       4  |      13  |       18 |       313|       1  |        1 |        0 |       10 |        0 |         14|    27|
| ArgM-MOD|    0|     2 |    0 |    0 |    0  |       1 |         0 |        0 |       0  |       0  |        0 |         1|       598|        0 |        0 |        0 |        0 |          0|    54|
| ArgM-NEG|    1|     2 |    1 |    0 |    0  |       2 |         0 |        0 |       0  |       0  |        0 |         2|       1  |      266 |        0 |        0 |        0 |          3|    13|
| ArgM-PNC|    0|     9 |   20 |    0 |    0  |       2 |         7 |        0 |       0  |       0  |        4 |         4|       0  |        0 |      119 |        0 |        0 |          5|    11|
| ArgM-PRD|    0|     9 |    7 |    0 |    0  |       9 |         0 |        0 |       0  |       2  |        3 |        13|       0  |        0 |        2 |       74 |        2 |          1|    12|
| ArgM-REC|    0|     0 |    1 |    0 |    0  |       0 |         0 |        0 |       0  |       0  |        1 |         0|       0  |        0 |        1 |        1 |      105 |          0|     0|
| ArgM-TMP|    0|     4 |   13 |    1 |    1  |      45 |         7 |        0 |       3  |      13  |       31 |        24|       1  |        3 |        4 |        0 |        0 |        862|    32|
|       _ |   81|   138 |   24 |    0 |    1  |      37 |         9 |        2 |      13  |       3  |       27 |        28|       50 |        15|         8|         7|         0|         30| 38234|


|             | precision  |  recall  |f1-score   |support|
|-------------|-----------:|---------:|----------:|------:|
|        Arg0 |      0.83  |    0.91  |    0.87   |   1814|
|        Arg1 |      0.84  |    0.83  |    0.83   |   3138|
|        Arg2 |      0.51  |    0.49  |    0.50   |    707|
|        Arg3 |      0.56  |    0.08  |    0.14   |     65|
|        Arg4 |      0.55  |    0.51  |    0.53   |     55|
|    ArgM-ADV |      0.62  |    0.59  |    0.60   |    528|
|    ArgM-CAU |      0.67  |    0.56  |    0.61   |    170|
|    ArgM-DIR |      0.44  |    0.52  |    0.48   |     58|
|    ArgM-DIS |      0.87  |    0.59  |    0.71   |    543|
|    ArgM-EXT |      0.50  |    0.54  |    0.52   |     87|
|    ArgM-LOC |      0.66  |    0.77  |    0.71   |    676|
|    ArgM-MNR |      0.61  |    0.61  |    0.61   |    516|
|    ArgM-MOD |      0.92  |    0.91  |    0.91   |    656|
|    ArgM-NEG |      0.90  |    0.91  |    0.91   |    291|
|    ArgM-PNC |      0.68  |    0.66  |    0.67   |    181|
|    ArgM-PRD |      0.67  |    0.55  |    0.61   |    134|
|    ArgM-REC |      0.94  |    0.96  |    0.95   |    109|
|    ArgM-TMP |      0.84  |    0.83  |    0.83   |   1044|
|           _ |      0.99  |    0.99  |    0.99   |  38707|
|             |            |          |           |       |
|    accuracy |            |          |    0.94   |  49479|
|   macro avg |      0.71  |    0.67  |    0.68   |  49479|
|weighted avg |      0.94  |    0.94  |    0.94   |  49479|


# References

1. [Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/abs/1703.04826)
2. [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115.pdf)
3. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
4. [GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs](https://arxiv.org/abs/1803.07294)
5. [Look Again at the Syntax: Relational Graph Convolutional Network for Gendered Ambiguous Pronoun Resolution](https://arxiv.org/abs/1905.08868) [code](https://github.com/ianycxu/RGCN-with-BERT)
6. [Deep Graph Library](https://www.dgl.ai)
7. [The CoNLL-2009 shared task: syntactic and semantic dependencies in multiple languages](https://dl.acm.org/doi/10.5555/1596409.1596411)
8. [2009 Shared task evaluation script](https://ufal.mff.cuni.cz/conll2009-st/scorer.html)
9. [Super-Convergence: Very Fast Training of NeuralNetworks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
10. [Label Distribution Learning](https://arxiv.org/abs/1408.6027)
11. [On Loss Functions for Deep Neural Networks in Classification](https://arxiv.org/abs/1702.05659)
12. [Training Products of Experts by Minimizing Contrastive Divergence](https://www.mitpressjournals.org/doi/10.1162/089976602760128018)
13. [Selecting weighting factors in logarithmic opinion pools.pdf](https://dl.acm.org/doi/10.5555/3008904.3008942)

#!/bin/bash
mkdir -p models

# Download srl model
if [ ! -f models/srl.pt ]; then
  wget https://surfdrive.surf.nl/files/index.php/s/kOgUm0oEpmx5HiZ/download -O models/srl.pt
fi

# Download word vectors
if [ ! -f models/fasttext.model.bin ]; then
  wget https://surfdrive.surf.nl/files/index.php/s/085yxFcRmn0osMw/download -O models/fasttext.model.bin
fi

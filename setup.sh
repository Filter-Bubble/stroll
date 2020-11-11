#!/bin/bash
mkdir -p models

if [ ! -f models/srl.pt ]; then
  wget https://surfdrive.surf.nl/files/index.php/s/kOgUm0oEpmx5HiZ/download -O models/srl.pt
fi

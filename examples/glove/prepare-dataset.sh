#! /bin/bash

mkdir -p dataset
cd dataset
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
cd ..
./convert.py 100 dataset/glove.twitter.27B.100d.txt

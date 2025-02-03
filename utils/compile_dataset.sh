#!/bin/bash
# This script takes one complete dataset from the dataset crawler and packs it
# in a zip file. It should be executed from the same directory as the crawler.
# Usage:
#   ./compile_dataset.sh <dataset-name> <dataset-index>

mkdir -p $1-data/raw/

cp data/graph_labels.npy $1-data/raw/
cp data/A_$2.txt $1-data/raw/A.txt
cp data/new_bert_feature_$2.npz $1-data/raw/new_bert_feature.npz
cp data/new_spacy_feature_$2.npz $1-data/raw/new_spacy_feature.npz
cp data/new_profile_feature_$2.npz $1-data/raw/new_profile_feature.npz
cp data/new_content_feature_$2.npz $1-data/raw/new_content_feature.npz
cp data/node_graph_id_$2.npy $1-data/raw/node_graph_id.npy
cp data/test_idx_$2.npy $1-data/raw/test_idx.npy
cp data/train_idx_$2.npy $1-data/raw/train_idx.npy
cp data/val_idx_$2.npy $1-data/raw/val_idx.npy

cd $1-data && zip -r $1-mumin.zip *

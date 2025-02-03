#!/bin/bash
# To run all baseline models, execute this script from within GNN-FakeNews/gnn_model.
# Create a folder out within the gnn_model folder before running the script!

features=( "bert" "spacy" "content" "profile" )
datasets=( "politifact" "gossipcop" "mumin-6h" "mumin-14h" )
models=( "gcn" "gat" "sage" )

for i in "${models[@]}"
do
	for j in "${datasets[@]}"
	do
		for k in "${features[@]}"
		do
			python3 gnn.py --model=$i --dataset=$j --feature=$k | grep "Test set" > out/$i-$j-$k.txt
		done
	done
done

#!/bin/bash
# To run all GCN models, execute this script from within GNN-FakeNews/gnn_model.
# Create a folder out within the gnn_model folder before running the script!

features=( "bert" "spacy" "content" "profile" )
datasets=( "politifact" "gossipcop" "mumin-6h" "mumin-14h" )
models=( "gcnfn.py" "bigcn.py" "gnncl.py" )

for i in "${models[@]}"
do
	for j in "${datasets[@]}"
	do
		for k in "${features[@]}"
		do
			python3 $i --dataset=$j --feature=$k | grep "Test set" > out/$i-$j-$k.txt
		done
	done
done

# Dataset

A dataset created by the Crawler is comprised of ten files containing all of the information regarding Tweet cascades and features. The files and their descriptions are as follows:

## A.txt
A text document containing adjacent node IDs on each line.

## graph_labels.npy
A NumPy array of length n composed of true (0) and false (1) indicators, where the index of each truth indicator corresponds with the graph ID to which it belongs.

## new_bert_features.npz
A NumPy matrix containing data corresponding to text content, classified with BERT. The index of each row corresponds to the node ID, while each row contains the data for its node.

## new_content_features.npz
A NumPy matrix containing data corresponding to text content. The index of each row corresponds to the node ID, while each row contains the data for its node.

## new_spacy_features.npz
A NumPy matrix containing data corresponding to text content, classified with spaCy. The index of each row corresponds to the node ID, while each row contains the data for its node.

## node_graph_id.npy
A NumPy array of length n, where n is the maximum node ID. The index of an element is the node ID, and the element is the ID of the graph to which the node belongs. 

## test_idx.npy
A NumPy array containing a random selection of 70% of the graph IDs.

## train_idx.npy
A NumPy array containing a random selection of 20% of the graph IDs.

## val_idx.npy
A NumPy array containing a random selection of 10% of the graph IDs.

# Raw Data

## Adjacency_list_x.pickle
A pickle file containing a NumPy array composed of arrays containing adjacent node IDs.

## Tweet_df_x.pickle
A pickle file containing a Pandas dataframe of Twitter data, specifically Tweet objects, from the MuMin dataset.

## User_df_x.pickle
A pickle file containing a Pandas dataframe of Twitter data, specifically User objects, from the MuMin dataset.

import numpy as np
from online_extract import *
import scipy.sparse
from collections import Counter

def reorganize_dump_all(tweet_df,claim_df,adjacency_list,spacy_vecs,bert_vecs,profile_vecs,content_vecs,iter_idx):

    labels = list(set(tweet_df['graph_label']))

    ## fix graph nums - do not consider singular graphs, reorder graph nums
    old_new_graph_nums = {}
    count_graph_labels = Counter(tweet_df['graph_label'])
    new_old_graph_nums = {}
    j = 0
    for i in range(len(labels)):
        if count_graph_labels[labels[i]] == 1:
            continue
        old_new_graph_nums[labels[i]] = j
        new_old_graph_nums[j] = labels[i]
        j+=1


    next_id=0
    old_new_idx = {}
    for graph_num in labels: #range(np.int64(max(tweet_df['graph_label']))+1):
        if count_graph_labels[graph_num] == 1:
            continue
        for node_num in range(len(tweet_df)):
            if tweet_df['graph_label'].iloc[node_num] == graph_num:
                old_new_idx[node_num] = next_id #old:new
                next_id+=1


    graph_labels = np.zeros(len(old_new_graph_nums), dtype=np.int64)
    graph_ind = 0
    for i in range(len(labels)):
        if labels[i] not in old_new_graph_nums:
            continue
        if claim_df['label'][int(labels[i])] != "factual":
            graph_labels[graph_ind] = 1
            graph_ind += 1
    with open(f"data/graph_labels_{iter_idx}.npy", "wb") as f:
        np.save(f,graph_labels)

    # save A
    save_adjacency_list(adjacency_list,tweet_df,old_new_idx,iter_idx)

    # save node graph id
    save_node_graph_id(tweet_df, iter_idx, old_new_idx, old_new_graph_nums)

    # save features 
    spacy_vecs_correct = np.zeros((len(old_new_idx),spacy_vecs.shape[1]))
    for i in range(len(spacy_vecs)):
        if i not in old_new_idx:
            continue
        spacy_vecs_correct[old_new_idx[i],:] = spacy_vecs[i,:]
    spacy_vecs_correct = scipy.sparse.csr_matrix(spacy_vecs_correct)
    save_features(spacy_vecs_correct,iter_idx,"spacy")
    
    bert_vecs_correct = np.zeros((len(old_new_idx),bert_vecs.shape[1]))
    for i in range(len(bert_vecs)):
        if i not in old_new_idx:
            continue
        bert_vecs_correct[old_new_idx[i],:] = bert_vecs[i,:]
    bert_vecs_correct = scipy.sparse.csr_matrix(bert_vecs_correct)
    save_features(bert_vecs_correct,iter_idx,"bert")

    profile_vecs_correct = np.zeros((len(old_new_idx),profile_vecs.shape[1]))
    for i in range(len(profile_vecs)):
        if i not in old_new_idx:
            continue
        profile_vecs_correct[old_new_idx[i],:] = profile_vecs[i,:]
    profile_vecs_correct = scipy.sparse.csr_matrix(profile_vecs_correct)
    save_features(profile_vecs_correct,iter_idx,"profile")

    content_vecs_correct = np.zeros((len(old_new_idx),content_vecs.shape[1]))
    for i in range(len(content_vecs)):
        if i not in old_new_idx:
            continue
        content_vecs_correct[old_new_idx[i],:] = content_vecs[i,:]
    content_vecs_correct = scipy.sparse.csr_matrix(content_vecs_correct)
    save_features(content_vecs_correct,iter_idx,"content")

    # save train, val, test
    save_train_val_test(tweet_df,len(old_new_graph_nums),count_graph_labels,new_old_graph_nums,iter_idx)

def save_adjacency_list(A,tweet_df,old_new_idx,idx):
    with open(f"data/A_{idx}.txt","w") as f:
        for a,b in A:
            idx1 = tweet_df.index.get_loc(tweet_df[tweet_df.tweet_id == a].iloc[-1].name)
            idx2 = tweet_df.index.get_loc(tweet_df[tweet_df.tweet_id == b].iloc[-1].name)
            #idx1 = tweet_df.index[tweet_df['tweet_id'] == a].tolist()[0]
            #idx2 = tweet_df.index[tweet_df['tweet_id'] == b].tolist()[0]
            new_idx1 = old_new_idx[idx1]
            new_idx2 = old_new_idx[idx2]
            f.write(f"{new_idx1}, {new_idx2}\n")

def save_node_graph_id(tweet_df,idx, old_new_idx, old_new_graph_nums):
    node_graph_id = np.zeros((len(old_new_idx)))
    for i in range(len(tweet_df)):
        graph_label = tweet_df['graph_label'].iloc[i]
        if graph_label not in old_new_graph_nums:
            continue
        node_graph_id[old_new_idx[i]] = int(old_new_graph_nums[graph_label])

    with open(f"data/node_graph_id_{idx}.npy", "wb") as f:
        np.save(f,node_graph_id)

def save_train_val_test(tweet_df,labels_length,count_graph_labels,new_old_graph_nums,idx):
    s = set()
    n = len(tweet_df)
    train_idx = []
    val_idx = []
    test_idx = []
    train_len = 0.2*n
    val_len = 0.1*n
    test_len = 0.7*n 
    arr = np.arange(0,labels_length)

    count_val = 0
    while(count_val < val_len):
        choice = np.random.choice(arr)
        while choice in s:
            choice = np.random.choice(arr)
        s.add(choice)
        count_val+=count_graph_labels[new_old_graph_nums[choice]] #len(tweet_df[tweet_df['graph_label'] == choice])
        val_idx.append(choice)

    count_test = 0
    while(count_test < test_len):
        choice = np.random.choice(arr)
        while choice in s:
            choice = np.random.choice(arr)
        s.add(choice)
        count_test+=count_graph_labels[new_old_graph_nums[choice]]#len(tweet_df[tweet_df['graph_label'] == choice])
        test_idx.append(choice)

    for label in arr:
        if label not in s:
            train_idx.append(label)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)

    with open(f"data/train_idx_{idx}.npy", "wb") as f:
        np.save(f,train_idx)
    with open(f"data/test_idx_{idx}.npy", "wb") as f:
        np.save(f,test_idx)
    with open(f"data/val_idx_{idx}.npy", "wb") as f:
        np.save(f,val_idx)

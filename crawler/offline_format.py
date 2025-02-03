import numpy as np

def transfrom_node_graph_id(tweet_df, idx):
    node_graph_id = np.zeros((len(tweet_df)))
    labels = list(set(tweet_df['graph_label']))
    d = {}
    for i in range(len(labels)):
        d[labels[i]] = i

    for i in range(len(tweet_df)):
        graph_label = tweet_df['graph_label'][i]
        node_graph_id[i] = d[graph_label]

    with open(f"data/node_graph_id_{idx}.npy", "wb") as f:
        np.save(f,node_graph_id)

def transfrom_graph_labels(claim_df, tweet_df, idx):
    labels = list(set(tweet_df['graph_label']))
    d = {}
    for i in range(len(labels)):
        d[i] = labels[i]

    graph_labels = np.zeros(len(labels))

    for i in range(len(labels)):
        if claim_df['label'][int(d[i])] == "factual":
            graph_labels[i] = 1

    with open(f"data/graph_labels_{idx}.npy", "wb") as f:
        np.save(f,graph_labels)

def transfrom_train_val_test(tweet_df, claim_df, idx):
    labels = list(set(tweet_df['graph_label']))
    d = {}
    for i in range(len(labels)):
        d[i] = labels[i]

    s= set()
    n = len(tweet_df)
    train_idx = []
    val_idx = []
    test_idx = []
    train_len = 0.2*n
    val_len = 0.1*n
    test_len = 0.7*n
    arr = np.arange(0,len(labels))

    count_val = 0
    while(count_val < val_len):
        choice = np.random.choice(arr)
        while choice in s:
            choice = np.random.choice(arr)
        s.add(choice)
        count_val+=len(tweet_df[tweet_df['graph_label'] == d[choice]])
        val_idx.append(choice)

    count_test = 0
    while(count_test < test_len):
        choice = np.random.choice(arr)
        while choice in s:
            choice = np.random.choice(arr)
        s.add(choice)
        count_test+=len(tweet_df[tweet_df['graph_label'] == d[choice]])
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

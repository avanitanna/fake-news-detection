import random
import numpy as np
import scipy.sparse
from collections import OrderedDict

# creates dictionary->{graph id : [node ids]}
def create_graphs_dict(node_graph_id_path):
    data_array = np.load(node_graph_id_path)
    x=0
    prev = ''
    graphs_dict = dict()
    for graph_item in data_array: 
        #print(f"Node ID:{x}, Graph ID: {graph_item}")
        graph_item = str(graph_item)
        if graph_item == prev:
            graphs_dict[graph_item].append(x)
        else:
            graphs_dict[graph_item] = [x]
        prev = graph_item
        x+=1
    return graphs_dict

def create_new_graph_ids(graph_dict):
    new_ids = list()
    for x in range(0,len(graph_dict)):
        new_ids.append(x)  
    return new_ids

# gives dictionary values a new random key from the list of graph ids
# creates dictionary of old+new graph ids->{old graph ID : new ID}
def rearrange_graph(graph_dict):
    print("> Assigning new graph IDs...")
    new_graph = dict()
    g_id = dict()
    new_ids = create_new_graph_ids(graph_dict)
    for id in graph_dict:
        rand_id = random.choice(new_ids)
        new_ids.remove(rand_id)
        new_graph[rand_id] = graph_dict[id]
        g_id[id] = rand_id
    return new_graph,g_id

# sorts dictionary by key value and removes removed graphs
def key_sort(graph_dict, removed_graphs):
    new_graph = dict()
    for i in sorted (graph_dict):
        if i in removed_graphs:
            continue
        else:
            new_graph[i] = graph_dict[i]
    return new_graph

# creates dict of node ids->{old node id : new node id}
# replaces old node ids in graph_dict
def relabel_nodes(graph_dict):
    print("> Relabeling nodes...")
    node_dict = dict()
    new_id = 0
    for graph_id in graph_dict:
        for i in range(0,len(graph_dict[graph_id])):
            node_dict[graph_dict[graph_id][i]] = new_id
            graph_dict[graph_id][i] = new_id
            new_id+=1
    return graph_dict, node_dict

# creates numpy array (like node_graph_id.npy) with new graph + node id's
# index is node id, value is the graph id to which the node belongs 
# creates dictionary of graph ids with counts of nodes
def create_graph_array(graph_dict, nodes):
    print("> Creating node_graph_id array...")
    ng_count = {}
    node_graph = [0] * len(nodes)
    new_graph_id=0
    for graph_id in graph_dict:
        ng_count[new_graph_id]=0
        for node_id in graph_dict[graph_id]:
            node_graph[int(node_id)] = new_graph_id
            ng_count[new_graph_id]+=1
        new_graph_id+=1
    return node_graph,ng_count

# given graph id dict ({old:new})
# returns array of graph labels on new ids
def create_labels(id_dict, graph_labels_path):
    data_array = np.load(graph_labels_path)
    labels = [0] * len(data_array)
    for i in np.arange(0.0,len(labels)): 
        # switch the conditional to "== 0" if using "graph_labels_wrong"    
        if data_array[int(i)] == 1:
            labels[id_dict[str(i)]] = 0 
        else:
            labels[id_dict[str(i)]] = 1
        #print(f'Old ID: {i}, New ID: {id_dict[str(i)]}, Value: {labels[id_dict[str(i)]]}')    
    return labels

def get_diff(labels):
    zero = 0
    one = 0
    for i in range(0,len(labels)):
        # switch the conditional to "== 0" if using "graph_labels_wrong"    
        if labels[i] == 1:
            one+=1
        else:
            zero+=1
    print(f"Real Count = {zero}")
    print(f"Fake Count = {one}")
    return (one - zero)

# cuts dataset to have the same amount of zeros as ones
# returns newly cut dataset, array of removed graph ids
def cut_dataset(labels):
    print("> Balancing dataset...")
    removed_indexes = []
    diff = get_diff(labels)
    index = len(labels) - 1
    while diff > 0:
        # switch the conditional to "== 0" if using "graph_labels_wrong"    
        if labels[index] == 1:
            labels.pop(index)
            removed_indexes.append(index)
            diff-=1
        index-=1
    print(">> Post balance:")
    get_diff(labels)
    return labels, removed_indexes

def save_features(name,vec):
    scipy.sparse.save_npz(f"new_{name}_feature.npz",vec) 

def reassign_features(nodes, bert_path, content_path, profile_path, spacy_path):
    print("> Reassigning feature matrixes...")
    bert = scipy.sparse.load_npz(bert_path)
    bert_vecs = bert.toarray()
    bert_vecs_correct = np.zeros((len(nodes),bert_vecs.shape[1]))
    
    content = scipy.sparse.load_npz(content_path)
    content_vecs = content.toarray()
    content_vecs_correct = np.zeros((len(nodes),content_vecs.shape[1]))
    
    profile = scipy.sparse.load_npz(profile_path)
    profile_vecs = profile.toarray()
    profile_vecs_correct = np.zeros((len(nodes),profile_vecs.shape[1]))

    spacy = scipy.sparse.load_npz(spacy_path)
    spacy_vecs = spacy.toarray()
    spacy_vecs_correct = np.zeros((len(nodes),spacy_vecs.shape[1]))
    
    for i in range(len(spacy_vecs)):
        if i not in nodes:
            continue
        spacy_vecs_correct[nodes[i],:] = spacy_vecs[i,:]
        bert_vecs_correct[nodes[i],:] = bert_vecs[i,:]
        profile_vecs_correct[nodes[i],:] = profile_vecs[i,:]
        content_vecs_correct[nodes[i],:] = content_vecs[i,:]

    spacy_vecs_correct = scipy.sparse.csr_matrix(spacy_vecs_correct)
    bert_vecs_correct = scipy.sparse.csr_matrix(bert_vecs_correct)
    profile_vecs_correct = scipy.sparse.csr_matrix(profile_vecs_correct)
    content_vecs_correct = scipy.sparse.csr_matrix(content_vecs_correct)
    save_features("spacy",spacy_vecs_correct)
    save_features("bert",bert_vecs_correct)
    save_features("profile",profile_vecs_correct)
    save_features("content",content_vecs_correct)

def create_adj_list(nodes, a_path):
    print("> Creating adjacency list...")
    newnodes = {}
    removed_nodes = []
    with open(a_path,"r") as f:
        for node in f:
            oldnodes = node.strip().split(",")
            try:
                newnodes[nodes[int(oldnodes[1])]] = nodes[int(oldnodes[0])]
            except KeyError:
                removed_nodes.append(int(oldnodes[1]))
        ordered = OrderedDict(sorted(newnodes.items(), key=lambda t:t[0]))
        with open('A_new.txt', 'w') as t:
            for val in ordered:
                #print(f"{ordered[val]},{val}\n")
                t.write(f"{ordered[val]},{val}\n")
    f.close()
    return removed_nodes

def save_train_val_test(node_graph_list,labels_length,count_graph_labels):
    print("> Saving train/val/test...")
    s = set()
    n = len(node_graph_list)
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
        count_val+=count_graph_labels[choice] #len(tweet_df[tweet_df['graph_label'] == choice])
        val_idx.append(choice)

    count_test = 0
    while(count_test < train_len):
        choice = np.random.choice(arr)
        while choice in s:
            choice = np.random.choice(arr)
        s.add(choice)
        count_test+=count_graph_labels[choice]#len(tweet_df[tweet_df['graph_label'] == choice])
        train_idx.append(choice)

    for label in arr:
        if label not in s:
            test_idx.append(label)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)

    with open(f"train_idx.npy", "wb") as f:
        np.save(f,train_idx)
    with open(f"test_idx.npy", "wb") as f:
        np.save(f,test_idx)
    with open(f"val_idx.npy", "wb") as f:
        np.save(f,val_idx)

# if using graph_labels_wrong.npy -> change conditionals in create_labels(), cut_dataset(), and get_diff()
PATHS = {'node_graph_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/GNN-FakeNews/data/raw/node_graph_id.npy',
        'graph_labels_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/GNN-FakeNews/data/raw/graph_labels.npy',
        'bert_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/GNN-FakeNews/data/raw/new_bert_feature.npz',
        'content_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/GNN-FakeNews/data/raw/new_content_feature.npz',
        'profile_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/GNN-FakeNews/data/raw/new_profile_feature.npz',
        'spacy_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/GNN-FakeNews/data/raw/new_spacy_feature.npz',
        'a_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/GNN-FakeNews/data/raw/A.txt'}

print(">>> Initializing dataset balancer...")
og_graphs_dict = create_graphs_dict(PATHS['node_graph_path'])
new_graph, g_id = rearrange_graph(og_graphs_dict)
relabels = create_labels(g_id,PATHS['graph_labels_path'])
cut_labels, removed_graphs = cut_dataset(relabels)
labels = np.array(cut_labels)
np.save('new_graph_labels',labels)
sorted_graph = key_sort(new_graph, removed_graphs)
re_graph, nodes = relabel_nodes(sorted_graph)
removed_nodes = create_adj_list(nodes, PATHS['a_path'])
reassign_features(nodes,PATHS['bert_path'],PATHS['content_path'],PATHS['profile_path'],PATHS['spacy_path'])
np_node_graph_id,ng_count = create_graph_array(re_graph,nodes)
np.save('new_node_graph_id',np.array(np_node_graph_id))
save_train_val_test(np_node_graph_id,len(cut_labels),ng_count)
print(">>> Complete!")
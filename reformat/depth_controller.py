import csv
from operator import indexOf
import numpy as np
import scipy.sparse

FILE_PATHS = {'adjacency_list':"/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/raavi_data/gossipcop/A.txt",
            'node_graph_id':"/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/raavi_data/gossipcop/node_graph_id.npy",
            'graph_labels_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/raavi_data/gossipcop/graph_labels.npy',
            'bert_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/raavi_data/gossipcop/new_bert_feature.npz',
            'content_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/raavi_data/gossipcop/new_content_feature.npz',
            'profile_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/raavi_data/gossipcop/new_profile_feature.npz',
            'spacy_path':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/raavi_data/gossipcop/new_spacy_feature.npz',
            'csv':""}

# creates list of adjacencies
a_file = open(FILE_PATHS["adjacency_list"],"r")
file_stripped = a_file.read().strip().split("\n")
ADJ = []

ng = np.load(FILE_PATHS["node_graph_id"])
NODE_GRAPH = ng.tolist() 

# creates dict {graph id : [adjacency vectors]}
GRAPHID_ADJ = dict()
for x in file_stripped:
    o = x.split(",")
    ADJ.append(o)
    try:
        if NODE_GRAPH[int(o[0])] not in GRAPHID_ADJ:
            GRAPHID_ADJ[NODE_GRAPH[int(o[0])]] = [o]
        else:
            GRAPHID_ADJ[NODE_GRAPH[int(o[0])]].append(o)
    except:
        continue

#for i in GRAPHID_ADJ:
#    print(f"Graph ID: {i}, Nodes: {GRAPHID_ADJ[i]}\n")

# given node_graph_id list
# returns {graph id: [node ids]} and {graph id:node count}
def create_graph_dicts(node_graph_id):
    node_graph = dict()
    graph_node_count = dict()
    for node_id in range(len(node_graph_id)):
        graph_id = int(node_graph_id[node_id])
        if graph_id not in graph_node_count:
            graph_node_count[int(node_graph_id[node_id])] = 0
        if graph_id not in node_graph:
            node_graph[int(node_graph_id[node_id])] = []
        node_graph[int(node_graph_id[node_id])].append(node_id)
        graph_node_count[int(node_graph_id[node_id])]+=1
    return node_graph, graph_node_count


# returns a list of all possible next vectors
# node = v1, has_next = [[v1,v2],[v1,v3],...,[v1,vN]]
def has_next(node, adjacency_list):
    nexts = []
    for n in adjacency_list:
        if node == n[0]:
            nexts.append(n)
    return nexts


# returns a list of all possible previous vectors
# node = v1, has_prev = [[v0,v1],[v00,v1],...,[vN,v1]]
def has_prev(node, adjacency_list):
    nexts = []
    for n in adjacency_list:
        if node == n[1].strip():
            nexts.append(n)
    return nexts


# removes vectors with EoB
# [[v1,v2],[v1,v3,'EoB'],[v2,v4]] -> [[v1,v2],[v2,v4]]
def remove_zeros(l):
    new_l = []
    for x in l:
        if 'EoB' not in x:
            new_l.append(x)
    return new_l


# given a layers dictionary and a max depth, 
# returns a layers dictionary with everything above max depth cut off
# and a list of removed node ID's
def depth_cutter(layers_dict,max_depth):
    nlayer = {}
    removed_nodes = []
    keys = list(layers_dict.keys())
    if max_depth in keys:
        for l in range(1,max_depth):
            nlayer[l] = layers_dict[l].copy()
        for i in range(max_depth,len(layers_dict)):
            for vector in layers_dict[i]:
                if int(vector[0]) not in removed_nodes:
                    removed_nodes.append(int(vector[0]))
                if int(vector[1]) not in removed_nodes:
                    removed_nodes.append(int(vector[1]))
    return nlayer, removed_nodes


# given a layers dict
# creates list of adjacencies and appends them to a text file
def update_adj_list(adj_list):
    with open('A_new.txt', 'a') as t:
        for vector in adj_list:
            #print(f"{vector[0]},{vector[1]}\n")
            t.write(f"{vector[0]},{vector[1]}\n")
    t.close()


# given a graph id and its corresponding layers dict
# returns array of its section of node_graph_id 
def create_ng_section(graph_id, layers_dict):
    ng = []
    ln = []
    for layer in layers_dict:
        for vector in layers_dict[layer]:
            if int(vector[0]) not in ln:
                ln.append(int(vector[0]))
            if int(vector[1]) not in ln:
                ln.append(int(vector[1]))
    ng = []*len(ln)
    for i in range(len(ln)):
        ng.append(graph_id)
    return ng


# returns the layer the vector is in, if in a layer
# else returns false
def in_branch(vector,layer_dict):
    for i in layer_dict.keys():
        if vector in layer_dict[i]:
            return i
    return 1


# given a graph id
# returns depth of layers, layers dict: {layer (1..n) : [adj vec at layer]}
def get_layers(graph_id):
    cur_layer = 1 
    layer = {1 : []}
    for i in range(len(GRAPHID_ADJ[graph_id])):   
        cur_vec = GRAPHID_ADJ[graph_id][i]
        cur_layer = in_branch(cur_vec,layer)
        connector = cur_vec[1].strip()
        if cur_layer == 1:
            layer[1].append(cur_vec)
        if has_next(connector,GRAPHID_ADJ[graph_id]) == []:
            continue
        else:
            if cur_layer+1 not in layer.keys():
                layer[cur_layer+1] = []
            for addition in has_next(connector,GRAPHID_ADJ[graph_id]):
                layer[cur_layer+1].append(addition)
    return max(list(layer.keys()))+1, layer


# given a layers dict
# returns the maximum depth path of the graph
def max_depth(layers):
    #print(layers)
    max_depth = []
    depths = list(layers.keys())
    depths.reverse()
    depths.pop()
    prev_vec = []
    for i in depths:
        if prev_vec == []:
            prev_vec.append(layers[i][0])
        for vec in prev_vec:
            connector = vec[0].strip()
            if has_prev(connector,layers[i-1]) == []:
                vec.append('EoB')
                prev_vec = []
            else:
                max_depth = [vec] + max_depth
                prev_vec = has_prev(connector,layers[i-1])
                if i == 2:
                    max_depth = prev_vec + max_depth
                break
    return remove_zeros(max_depth)


# given a list of removed nodes,
# returns a node_graph_id list with those nodes removed
# returns a dictionary of old and new ids {old id: new id}
def node_remover(removed_nodes):
    n_graph = []
    oldnew = {} # {old id: new id}
    for node_id in range(len(NODE_GRAPH)):
        if node_id in removed_nodes:
            ad = {node_id : "REMOVED"}
            oldnew.update(ad)
        else:
            ad = {node_id : len(n_graph)}
            oldnew.update(ad)
            n_graph.append(NODE_GRAPH[node_id])
    finaloldnew = {}
    for i in oldnew:
        if oldnew[i] != "REMOVED":
            finaloldnew[i] = oldnew[i]
    return n_graph,finaloldnew


# helper for reassign_features
def save_features(name,vec):
    scipy.sparse.save_npz(f"new_{name}_feature.npz",vec) 


# takes node ids->{old node id : new node id}
# reassigns features
def reassign_features(nodes, bert_path, content_path, profile_path, spacy_path):
    print("Reassigning feature matrixes...")
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
        try:
            spacy_vecs_correct[nodes[i],:] = spacy_vecs[i,:]
            bert_vecs_correct[nodes[i],:] = bert_vecs[i,:]
            profile_vecs_correct[nodes[i],:] = profile_vecs[i,:]
            content_vecs_correct[nodes[i],:] = content_vecs[i,:]
        except:
            print(f"BAD INDEX: {i}")

    spacy_vecs_correct = scipy.sparse.csr_matrix(spacy_vecs_correct)
    bert_vecs_correct = scipy.sparse.csr_matrix(bert_vecs_correct)
    profile_vecs_correct = scipy.sparse.csr_matrix(profile_vecs_correct)
    content_vecs_correct = scipy.sparse.csr_matrix(content_vecs_correct)
    save_features("spacy",spacy_vecs_correct)
    save_features("bert",bert_vecs_correct)
    save_features("profile",profile_vecs_correct)
    save_features("content",content_vecs_correct)


# takes node_graph_id list, length of labels, and {graph id: node count}
# saves test data
def save_train_val_test(node_graph_list,labels_length,count_graph_labels):
    print("Saving train/val/test...")
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


def create_new_adj(oldnew):
    newADJ = []
    for vec in ADJ:
        try:
            v0 = oldnew[int(vec[0])]
            v1 = oldnew[int(vec[1])]
            new_vec = [v0,v1] 
            newADJ.append(new_vec)
        except:
            continue
    return newADJ


def main():

    # change this variable!
    MAX_DEPTH = 10

    removed_nodes = []
    for graph_id in GRAPHID_ADJ:
        count,l = get_layers(graph_id)
        nl,rn = depth_cutter(l,MAX_DEPTH)
        for n in rn:
            removed_nodes.append(n)
    node_graph_id,oldnew_node = node_remover(removed_nodes)
    print("Creating node_graph_id...")
    np.save('node_graph_id',np.array(node_graph_id))
    adj_list = create_new_adj(oldnew_node)
    update_adj_list(adj_list)
    ng_dict,graph_node_count = create_graph_dicts(node_graph_id)
    reassign_features(oldnew_node,FILE_PATHS["bert_path"],FILE_PATHS["content_path"],FILE_PATHS["profile_path"],FILE_PATHS["spacy_path"])
    labels = np.load(FILE_PATHS["graph_labels_path"])
    save_train_val_test(node_graph_id,len(labels),graph_node_count)
    print("> Complete!")


main()
#count, layers = get_layers(26)
#print(max_depth(layers))

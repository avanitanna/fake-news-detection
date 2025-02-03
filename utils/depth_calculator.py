import csv
from operator import indexOf
import numpy as np

FILE_PATHS = {'adjacency_list':"/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/Ag.txt",
            'node_graph_id':"/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/gnode_graph_id.npy",
            'csv':'/Users/rvi/Documents/rcPersonal/UCSB/fake-news-detection/gossipcop.csv'}

# creates list of adjacencies
a_file = open(FILE_PATHS["adjacency_list"],"r")
file_stripped = a_file.read().strip().split("\n")
a = []

# creates dict {index/node id : graph id}
ng = np.load(FILE_PATHS["node_graph_id"])
node_graph = dict()
for x in range(len(ng)):
    node_graph[x] = int(ng[x])

# creates dict {graph id : [adjacency vectors]}
graphn = dict()
for x in file_stripped:
    o = x.split(",")
    a.append(o)
    if node_graph[int(o[0])] not in graphn:
        graphn[node_graph[int(o[0])]] = [o]
    else:
        graphn[node_graph[int(o[0])]].append(o)


# returns a list of all possible next vectors
# node = v1, has_next = [[v1,v2],[v1,v3],...,[v1,vN]]
def has_next(node, adjacency_list):
    nexts = []
    for n in adjacency_list:
        if node == n[0]:
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


# calculates longest depth
def calc_depth(nodes):
    top_count = 0
    top_depth = []
    to_visit = []
    for current in nodes:   
        if current in top_depth:
            continue
        else:
            cur_count = 1
            # we append nodes that branch from our current node to to_visit
            to_visit.append(current)
            for node in to_visit:
                connector = node[1].strip()
                z = indexOf(to_visit,node)
                # if current node does not branch further and 
                # we've iterated to the end of to_visit, we have reached the end of the cascade 
                if has_next(connector,nodes) == [] and z == len(to_visit)-1:
                    # if cascade is bigger than current top, save it as top  
                    if cur_count > top_count:
                        top_count = cur_count
                        cur_count = 1
                        top_depth = to_visit.copy()
                    to_visit.clear()
                # if current node does not branch further
                # but there are other branches to explore
                elif has_next(connector,nodes) == []:
                    # the current node gets marked as an "End of Branch"
                    to_visit[z].append("EoB")
                    continue
                else:
                    # current node has further connections, add all potential branches to to_visit
                    cur_count+=1
                    for addition in has_next(connector,nodes):
                        to_visit.append(addition)
    # we run the top depth through remove_zeros() to remove all EoBs and have a continous depth
    return top_count,remove_zeros(top_depth)


ngraphid = dict()
for graphID in graphn:
    # this is the weirdest part of the script. 99% sure it works.
    # every time you run calc_depth, it gets rid of branches that don't continue.
    # after the fourth/fifth run, there are no further changes.
    c,d = calc_depth(graphn[graphID])
    t,x = calc_depth(d)
    b,q = calc_depth(x)
    e,w = calc_depth(q)
    j,h = calc_depth(w)
    ngraphid[graphID]={'top_depth':j+1,'depth_path':h}

f = open(FILE_PATHS["csv"], 'w')
writer = csv.writer(f)
writer.writerow(["graphID","top_depth","depth_path"])
for n in ngraphid:
    data = [n,ngraphid[n]['top_depth'],ngraphid[n]['depth_path']]
    writer.writerow(data)
f.close()
# !pip install mumin[all]==1.6.2 torchmetrics==0.7.2 --quiet
# !pip install dgl-cu111==0.7.2 -f https://data.dgl.ai/wheels/repo.html --quiet

## NOTE - need mumin-small.zip in the working directory 
from mumin import MuminDataset
dataset = MuminDataset(twitter_bearer_token="")
dataset.compile()
user_df = dataset.nodes['user'] 
print(user_df.head())

from datetime import datetime
import numpy as np
import pickle 
"""

Helper code to generate 10-dimensional user profile feature based on crawled user object using Twitter Developer API

"""

feature = np.zeros([len(user_df), 10], dtype=np.float32)
id_counter = 0
est_date = datetime.fromisoformat('2006-03-21') #CHANGE THIS!
for i in range(len(user_df)):
    # 1) Verified?, 2) Enable geo-spatial positioning, 3) Followers count, 4) Friends count
    vector = [int(user_df['verified'][i]), 0, user_df['num_followers'][i], user_df['num_followees'][i]]
    # 5) Status count, 6) Favorite count, 7) Number of lists
    vector += [user_df['num_tweets'][i], 0, user_df['num_listed'][i]]

    # 8) Created time (No. of months since Twitter established)
    user_date = user_df['created_at'][i] #datetime.strptime(user_df['created_at'][i], '%a %b %d %H:%M:%S +0000 %Y')
    month_diff = (user_date.year - est_date.year) * 12 + user_date.month - est_date.month
    vector += [month_diff]

    # 9) Number of words in the description, 10) Number of words in the screen name
    vector += [len(user_df['name'][i].split()), len(user_df['description'][i].split())]

    feature[id_counter, :] = np.reshape(vector, (1, 10))
    id_counter += 1
    print(id_counter)

f = open("profile_vecs.pickle","wb")
pickle.dump(feature,f)
f.close()


""" SPACY feature extraction 

from mumin import MuminDataset
dataset = MuminDataset(twitter_bearer_token="")
dataset.compile()
tweet_df = dataset.nodes['tweet'] 
#print(tweet_df.head())

#!pip install -U pip setuptools wheel
#!pip install -U spacy
#!python -m spacy download en_core_web_lg
#!python -m spacy validate
import spacy
import pickle 
# Load the installed model "en_core_web_lg"
nlp = spacy.load('en_core_web_lg')
spacy_vecs = []
for t in tweet_df['text']:
    spacy_vecs.append(nlp(t).vector)

f = open("spacy_vecs.pickle","wb")
pickle.dump(spacy_vecs,f)
f.close()
"""

""" BERT feature extraction 
pip install clip-server
pip install "clip-server[onnx]"
pip install clip-client
from clip_client import Client
c = Client('grpc://0.0.0.0')
# change in torch-flow.yml file - add with name as ViT-L/14 (model name) 
# https://clip-as-service.jina.ai/user-guides/server/
bert_vecs = c.encode(tweet_df['text'])
"""

""" SAVING npz features file as per politifact and gossipcop format
import pickle
f = pickle.load(open('spacy_vecs.pickle',"rb"))
import numpy as np
##  see what gossipcop features look like 
arr = np.load('C:/Users/Avani/Data/Learning/UCSB_PREP/CS292F_MLforGraphs/GNN-FakeNews-main/GNN-FakeNews-main/gnn_model/data/gossipcop/raw/new_spacy_feature.npz')
arr.files  
['indices', 'indptr', 'format', 'shape', 'data']
fmt = arr.f.format
idx = arr.f.indices
indptr = np.arange(len(f))*300
len(f)
4339
arr.f.shape
array([314262,    300], dtype=int64)
shp = np.array([4339,300], dtype=np.int64)
shp
array([4339,  300], dtype=int64)
data = np.concatenate(f)
len(data)/300
4339.0
fd = open("mumin_small_spacy_feature.npz", "wb")
np.savez(fd, indices = idx, indptr = indptr, format = fmt, shape = shp, data = data)

"""
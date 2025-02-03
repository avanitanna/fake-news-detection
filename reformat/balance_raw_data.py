import pickle
import random
import math
from collections import Counter
from mumin import MuminDataset
from clip_client import Client
import sys
import spacy
if not spacy.require_gpu(3):
    print("## Running on GPU 3 failed")
    sys.exit(1)
import pandas as pd
import numpy as np
import tweepy

print("## Load data")
dataset = MuminDataset(twitter_bearer_token="", size='large')
dataset.compile()
claim_df = dataset.nodes['claim']

with open("data_intermediary/user_df_37.pickle", "rb") as f:
    user_df = pickle.load(f)
with open("data_intermediary/tweet_df_37.pickle", "rb") as f:
    tweet_df = pickle.load(f)

print("## Loading done")
"""
print("## Get random sample")

# get all factual tweets in one df
true_tweet_df = pd.DataFrame(columns=["tweet_id","text","created_at",
    "is_retweet","graph_label","user_id"])
true_tweet_df = true_tweet_df.astype({"tweet_id": np.uint64, "text": object,
    "is_retweet": np.int64, "graph_label": np.int64, "user_id": np.uint64})

for i in range(len(tweet_df)):
    graph_label = int(tweet_df['graph_label'].iloc[i])
    tweet_id = tweet_df['tweet_id'].iloc[i]
    text = tweet_df['text'].iloc[i]
    created_at = tweet_df['created_at'].iloc[i]
    is_retweet = tweet_df['is_retweet'].iloc[i]
    user_id = tweet_df['user_id'].iloc[i]
    if claim_df['label'][graph_label] == "factual":
        tmp = pd.DataFrame([{'tweet_id' : tweet_id, 'text' : text, 'created_at' : created_at, 'is_retweet' : is_retweet,
                    'graph_label' : graph_label, 'user_id' : user_id}]).astype({'tweet_id':np.uint64,'text':object,
                    'created_at':'datetime64[ns]','is_retweet':np.int64,'graph_label':np.int64,'user_id':np.uint64})
        true_tweet_df = pd.concat([true_tweet_df, tmp],ignore_index=True)

true_labels = list(set(true_tweet_df['graph_label']))
num_graphs = len(list(set(tweet_df['graph_label'])))

# get a random sample of graph labes belonging to misinformation graphs that matches the size of factual graphs
random_false_labels = []
for i in range(len(true_labels)):
    l = random.randint(0, len(tweet_df)-1)
    while tweet_df['graph_label'].iloc[l] in true_labels or tweet_df['graph_label'].iloc[l] in random_false_labels:
        l = random.randint(0, len(tweet_df)-1)
    random_false_labels.append(tweet_df['graph_label'].iloc[l])

# get the tweets belonging to the labels
false_tweet_df = pd.DataFrame(columns=["tweet_id","text","created_at",
    "is_retweet","graph_label","user_id"])
false_tweet_df = false_tweet_df.astype({"tweet_id": np.uint64, "text": object,
    "is_retweet": np.int64, "graph_label": np.int64, "user_id": np.uint64})

for i in range(len(tweet_df)):
    graph_label = tweet_df['graph_label'].iloc[i]
    tweet_id = tweet_df['tweet_id'].iloc[i]
    text = tweet_df['text'].iloc[i]
    created_at = tweet_df['created_at'].iloc[i]
    is_retweet = tweet_df['is_retweet'].iloc[i]
    user_id = tweet_df['user_id'].iloc[i]
    if graph_label in random_false_labels:
        tmp = pd.DataFrame([{'tweet_id' : tweet_id, 'text' : text, 'created_at' : created_at, 'is_retweet' : is_retweet,
                'graph_label' : int(graph_label), 'user_id' : user_id}]).astype({'tweet_id':np.uint64,'text':object,
                'created_at':'datetime64[ns]','is_retweet':np.int64,'graph_label':np.int64,'user_id':np.uint64})
        false_tweet_df = pd.concat([false_tweet_df, tmp],ignore_index=True)


rand_tweet_df = pd.concat([true_tweet_df, false_tweet_df],ignore_index=True)
rand_tweet_df.reset_index()
with open("data_intermediary/rand_tweet_df.pickle", "wb") as f:
    pickle.dump(rand_tweet_df, f)

print("## Random sample is ready")
"""
rand_tweet_df = tweet_df
print("## Start pulling user timelines and extract features")

def correct_text(timeline):
    """ Remove special characters and URLs. """
    result = []
    special_characers = ['@', ',', '.', '-', '+', '/', '?', '#', '!', '$', '%', '^', '&', '*', '(', ')', '_', '=', '>', '<', '\\', '|', ']', '}', '[', '{', '\n', ':', ';', '\'', '"']
    for status in timeline:
        text = status.full_text
        t = text.split(" ")
        text = ""
        for i in range(len(t)):
            if not t[i].startswith("http"):
                text = " ".join([text, t[i]])
        for s in special_characers:
            text = " ".join(text.split(s))
        result.append(text)
    return result


consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
c = Client('grpc://0.0.0.0:51000')
nlp = spacy.load('en_core_web_lg')

user_feats = pd.DataFrame(columns=['user_id', 'bert_vec', 'spacy_vec'])
user_feats = user_feats.astype({'user_id': np.uint64, 'bert_vec': object,
    'spacy_vec': object})
graphs = set(rand_tweet_df['graph_label'])
print("### Extracting features for", len(graphs), "graphs")
i = 0
try:
    for g in graphs:
        print("#### Handling graph", i)
        tweets = rand_tweet_df[rand_tweet_df['graph_label'] == g]
        user_ids = set(tweets['user_id'])
        timelines = {}
        missing_ids = []
        existing = set(user_feats['user_id'])
        for uid in user_ids:
            if uid in existing:
                continue
            try:
                for page in tweepy.Cursor(api.user_timeline, user_id=uid, count=200, tweet_mode="extended").pages(1):
                    timelines[uid] = page
            except:
                missing_ids.append(uid)
        tweet_list = []
        for key in timelines:
            tweet_list.extend(timelines[key])
        if len(tweet_list) != 0:
            for uid in missing_ids:
                samp = random.choices(tweet_list, k=200)
                timelines[uid] = samp

        for uid in timelines:
            all_corrected = correct_text(timelines[uid])
            corrected = []
            for t in all_corrected:
                if not t.strip() == "":
                    corrected.append(t)
            if len(tweet_list) == 0 and len(corrected) == 0:
                continue
            while len(corrected) == 0:
                sampled = random.choices(tweet_list, k=200)
                all_corrected = correct_text(sampled)
                corrected = []
                for t in all_corrected:
                    if not t.strip() == "":
                        corrected.append(t)
                if len(corrected) == 0:
                    print("Again?!")
            all_texts = " ".join(corrected)
            spacy_vec = nlp(all_texts).vector
            b_vecs = c.encode(corrected)
            bert_vec = np.average(b_vecs, axis=0)
            df = pd.DataFrame([{'user_id': uid, 'bert_vec': bert_vec,
                'spacy_vec': spacy_vec}]).astype({'user_id': np.uint64,
                'bert_vec': object, 'spacy_vec': object})
            user_feats = pd.concat([user_feats, df], ignore_index=True)
        i += 1
except Error as e:
    print(e)
except Exception as e:
    print(e)

with open("data_intermediary/user_feats.pickle", "wb") as f:
    pickle.dump(user_feats, f)
print("## Timelines pulled, features extracted")
print("## Prepare for reorganize_dump_all")

# we have tweet_df and claim_df
# we ignore profile and content features
# we need adj list, bert feat, spacy feat
bert_vecs = []
spacy_vecs = []
for i in range(len(rand_tweet_df)):
    user_id = rand_tweet_df['user_id'].iloc[i]
    feats = user_feats[user_feats['user_id'] == user_id].iloc[0]
    spacy_vecs.append(feats['spacy_vec'])
    bert_vecs.append(feats['bert_vec'])
bert_vecs = np.array(bert_vecs)
spacy_vecs = np.array(spacy_vecs)

with open("data_intermediary/adjacency_list_37.pickle", "rb") as f:
    adj_list = pickle.load(f)
"""
new_adj_list = []
for i in range(len(adj_list)):
    tweet_1 = rand_tweet_df[rand_tweet_df['tweet_id'] == adj_list[i][0]]
    tweet_2 = rand_tweet_df[rand_tweet_df['tweet_id'] == adj_list[i][1]]
    if len(tweet_1) > 0 and len(tweet_2) > 0:
        new_adj_list.append(adj_list[i])
    if len(tweet_1) == 0 and len(tweet_2) > 0:
        print("Why do we only find tweet2?", i)
    if len(tweet_2) == 0 and len(tweet_1) > 0:
        print("Why do we only find tweet1?", i)
"""

import format_data
format_data.reorganize_dump_all(rand_tweet_df, claim_df, adj_list, spacy_vecs_np, bert_vecs_np, [], [], "correct_bert_spacy")

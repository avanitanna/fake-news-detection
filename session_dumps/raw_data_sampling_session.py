import pickle
with open("data_intermediary-small/user_df_76.pickle", "rb") as f:
    user_df = pickle.load(f)
len(user_df)
with open("data_intermediary-small/tweet_df_76.pickle", "rb") as f:
    tweet_df = pickle.load(f)
len(tweet_df)
tweet_df
from mumin import MuminDataset
dataset = MuminDataset(twitter_bearer_token="")
dataset.compile()
claim_df = dataset.nodes['claim']
claim_df
claim_df[0]
claim_df
claim_df['embedding'][0]
len(claim_df['embedding'][0])
history
clear
tweet_df
tweet_df.columns
import pandas as pd
true_tweet_df = pd.DataFrame(columns=["tweet_id","text","created_at","is_retweet","graph_label","user_id"])
claim_df[:][0]
claim_df.iloc[0]
tweet_df.columns
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
len(true_tweet_df)
from collections import Counter
count_graphs = Counter(true_tweet_df['graph_label'])
len(count_graphs)
count_users = Counter(true_tweet_df['user_id'])
len(count_users)
len(count_users) * 200
true_labels = list(set(true_tweet_df['graph_label']))
true_labels
random_false_labels = []
import math
math.random()
import random
num_graphs = len(list(set(tweet_df['graph_labels'])))
num_graphs = len(list(set(tweet_df['graph_label'])))
num_graphs
for i in range(len(true_labels)):
    l = random.randint(0, len(tweet_df)-1)
    while tweet_df['graph_label'].iloc[l] in true_labels:
        l = random.randint(0, len(tweet_df)-1)
    random_false_labels.append(tweet_df['graph_label'].iloc[l])
random_false_labels
random_false_labels = []
for i in range(len(true_labels)):
    l = random.randint(0, len(tweet_df)-1)
    while tweet_df['graph_label'].iloc[l] in true_labels or tweet_df['graph_label'].iloc[l] in random_false_labels:
        l = random.randint(0, len(tweet_df)-1)
    random_false_labels.append(tweet_df['graph_label'].iloc[l])
random_false_labels
len(list(set(random_false_labels)))
false_tweet_df = pd.DataFrame(columns=["tweet_id","text","created_at","is_retweet","graph_label","user_id"])
true_tweet_df
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
len(false_tweet_df)
len(list(set(false_tweet_df['user_id'])))
random_false_labels = []
for i in range(len(true_labels)):
    l = random.randint(0, len(tweet_df)-1)
    while tweet_df['graph_label'].iloc[l] in true_labels or tweet_df['graph_label'].iloc[l] in random_false_labels:
        l = random.randint(0, len(tweet_df)-1)
    random_false_labels.append(tweet_df['graph_label'].iloc[l])
false_tweet_df = pd.DataFrame(columns=["tweet_id","text","created_at","is_retweet","graph_label","user_id"])
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
len(list(set(false_tweet_df['user_id'])))
random_false_labels = []
for i in range(70):
    l = random.randint(0, len(tweet_df)-1)
    while tweet_df['graph_label'].iloc[l] in true_labels or tweet_df['graph_label'].iloc[l] in random_false_labels:
        l = random.randint(0, len(tweet_df)-1)
    random_false_labels.append(tweet_df['graph_label'].iloc[l])
false_tweet_df = pd.DataFrame(columns=["tweet_id","text","created_at","is_retweet","graph_label","user_id"])
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
len(list(set(false_tweet_df['user_id'])))
len(true_labels)
random_true_labels = random.sample(true_labels, 70)
rand_true_tweet_df = pd.DataFrame(columns=["tweet_id","text","created_at","is_retweet","graph_label","user_id"])
for i in range(len(tweet_df)):
    graph_label = int(tweet_df['graph_label'].iloc[i])
    tweet_id = tweet_df['tweet_id'].iloc[i]
    text = tweet_df['text'].iloc[i]
    created_at = tweet_df['created_at'].iloc[i]
    is_retweet = tweet_df['is_retweet'].iloc[i]
    user_id = tweet_df['user_id'].iloc[i]
    if graph_label in random_true_labels:
        tmp = pd.DataFrame([{'tweet_id' : tweet_id, 'text' : text, 'created_at' : created_at, 'is_retweet' : is_retweet,
                            'graph_label' : graph_label, 'user_id' : user_id}]).astype({'tweet_id':np.uint64,'text':object,
                            'created_at':'datetime64[ns]','is_retweet':np.int64,'graph_label':np.int64,'user_id':np.uint64})
        rand_true_tweet_df = pd.concat([rand_true_tweet_df, tmp],ignore_index=True)
len(list(set(rand_true_tweet_df['user_id'])))
len(list(set(rand_true_tweet_df['graph_label'])))
len(list(set(false_tweet_df['graph_label'])))
len(list(set(rand_true_tweet_df['user_id']).union(set(false_tweet_df['graph_label']))))
len(list(set(false_tweet_df['user_id'])))
rand_tweet_df = pd.concat([rand_true_tweet_df, false_tweet_df],ignore_index=True)


consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
import tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
timelines = pd.DataFrame(columns=['user_id', 'timeline'])
for uid in user_ids:
    try:
        for page in tweepy.Cursor(api.user_timeline, user_id=uid, count=200, tweet_mode="extended").pages(1):
            t = pd.DataFrame([{'user_id':uid, "timeline":page}])
            timelines = pd.concat([timelines, t], ignore_index=True)
    except:
        print("Could not get timeline for user_id", uid)
with open("data_intermediary-small/rand_users_timelines.pickle", "wb") as g:
    pickle.dump(timelines, g)
missing_ids = []
for uid in user_ids:
    if len(timelines[timelines['user_id'] == uid]) == 0:
        missing_ids.append(uid)
for uid in missing_ids:
    tweets = tweet_df[tweet_df['user_id'] == missing_ids[0]]
    if len(tweets) > 1:
        print(uid)
# -> no output!

for uid in missing_ids:
    tweets = rand_tweet_df[rand_tweet_df['user_id'] == uid]
    tweet_list = []
    for i in range(len(tweets)):
        graphs_tweets = tweet_df[tweet_df['graph_label'] == tweets['graph_label'].iloc[i]]
        user_set = set(graphs_tweets['user_id'])
        for user in user_set:
            tmp = list(timelines[timelines['user_id'] == user]['timeline'])
            if len(tmp) == 1:
                tweet_list.extend(tmp[0])
    samp = random.sample(tweet_list, 200)
    t = pd.DataFrame([{'user_id':uid, "timeline":samp}])
    missing_timeline_df = pd.concat([missing_timeline_df, t], ignore_index=True)
timelines = pd.concat([timelines, missing_timeline_df], ignore_index=True)
with open("data_intermediary-small/rand_users_timelines.pickle", "wb") as g:
    pickle.dump(timelines, g)


# let's do some feature extraction
# TODO import spacy and clip
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

bert_vecs = []
spacy_vecs = []

def random_sampling(uid):
    tweets = rand_tweet_df[rand_tweet_df['user_id'] == uid]
    tweet_list = []
    for i in range(len(tweets)):
        graphs_tweets = tweet_df[tweet_df['graph_label'] == tweets['graph_label'].iloc[i]]
        user_set = set(graphs_tweets['user_id'])
        for user in user_set:
            tmp = list(timelines[timelines['user_id'] == user]['timeline'])
            if len(tmp) == 1:
                tweet_list.extend(tmp[0])
    samp = random.sample(tweet_list, 200)
    return samp

for i in range(len(timelines)):
    all_corrected = correct_text(timelines['timeline'].iloc[i])
    corrected = []
    for t in all_corrected:
        if not t.strip() == "":
            corrected.append(t)
    while len(corrected) == 0:
        sampled = random_sampling(timelines['user_id'].iloc[i])
        all_corrected = correct_text(sampled)
        corrected = []
        for t in all_corrected:
            if not t.strip() == "":
                corrected.append(t)
        if len(corrected) == 0:
            print("Again?!")
    all_texts = " ".join(corrected)
    spacy_vecs.append(nlp(all_texts).vector)
    c = Client('grpc://0.0.0.0:51000')
    b_vecs = c.encode(corrected)
    bert_vecs.append(np.average(b_vecs, axis=0))

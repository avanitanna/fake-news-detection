import pickle
import random
import numpy as np
import spacy
from clip_client import Client
import scipy.sparse

print("## Load data")

with open("data_intermediary-small/tweet_df_76.pickle", "rb") as f:
    tweet_df = pickle.load(f)

with open("data_intermediary-small/rand_tweet_df_140_graphs.pickle", "rb") as f:
    rand_tweet_df = pickle.load(f)

with open("data_intermediary-small/rand_users_timelines.pickle", "rb") as g:
    timelines = pickle.load(g)

print("## Done loading data")

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

def correct_text(timeline):
#    """ Remove special characters and URLs. """
    result = []
    special_characers = ['@', ',', '.', '-', '+', '/', '?', '#', '!', '$', '%', '^', '&', '*', '(', ')', '_', '=', '>', '<', '\\', '|', ']', '}', '[', '{', '\n', ';', ':', '\'', '"']
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

nlp = spacy.load('en_core_web_lg')

spacy_vecs = []
bert_vecs = []
tweet_texts = []
flattened_tweet_texts = []

print("## Feature extraction, let's go!!")

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
    all_text = " ".join(corrected)
    spacy_vecs.append(nlp(all_text).vector)
    tweet_texts.append(corrected)
    flattened_tweet_texts.extend(corrected)
    #b_vecs = c.encode(corrected)
    #bert_vecs.append(np.average(b_vecs, axis=0))

spacy_features = np.array(spacy_vecs)
spacy_matrix = scipy.sparse.csr_matrix(spacy_features)
scipy.sparse.save_npz("data_intermediary-small/new_spacy_feature.npz", spacy_matrix)

print("## spacy features done, bert coming up")

# try to do it in 10 batches
with open("data_intermediary-small/tweet_texts_corrected.pickle", "wb") as f:
    pickle.dump(tweet_texts, f)

c = Client('grpc://0.0.0.0:51000')
with open("data_intermediary-small/tweet_texts_corrected.pickle", "rb") as f:
    tweet_texts = pickle.load(f)

timelines_per_req = len(tweet_texts) // 100
start = 0
end = timelines_per_req
bert_vecs = []

for i in range(100):
    print("Request", i)
    # get flattened array
    flattened = []
    for j in range(start, end):
        flattened.extend(tweet_texts[j])

    # call encode
    embeddings = c.encode(flattened)

    # deflatten array
    s, e = 0, 0
    for j in range(start, end):
        s = e
        e += len(tweet_texts[j])
        bert_vecs.append(np.average(embeddings[s:e], axis=0))

    # advance start and end
    start = end
    end += timelines_per_req
    if i == 98:
        end = len(tweet_texts)


bert_features = np.array(bert_vecs)
bert_matrix = scipy.sparse.csr_matrix(bert_features)
scipy.sparse.save_npz("data_intermediary-small/new_bert_feature.npz", bert_matrix)


## new ipython session, stitch the dataset together
import pickle
import numpy as np
import scipy.sparse_matrix
import scipy.sparse
bert_features = scipy.sparse.load_npz("data_intermediary-small/new_bert_feature.npz")
spacy_features = scipy.sparse.load_npz("data_intermediary-small/new_spacy_feature.npz")
with open("data_intermediary-small/rand_tweet_df_140_graphs.pickle", "rb") as f:
    rand_tweet_df = pickle.load(f)
from mumin import MuminDataset
dataset = MuminDataset(twitter_bearer_token="")
dataset.compile
dataset.compile()
claim_df = dataset.nodes['claim']
bert_features = bert_features.toarray()
spacy_features = spacy_features.toarray()
bert_vecs = []
with open("data_intermediary-small/rand_users_timelines.pickle", "rb") as f:
    rand_user_timelines = pickle.load(f)
rand_user_timelines.reset_index()
for i in range(len(rand_tweet_df)):
    # get userid
    user_id = rand_tweet_df['user_id'].iloc[i]
    # lookup index of userid in rand_user_timelines
    line_id = rand_user_timelines[rand_user_timelines['user_id'] == user_id].index[0]
    # add corresponding feature vector to bert_vecs
bert_vecs.append(bert_features[line_id])
spacy_vecs = []
for i in range(len(rand_tweet_df)):
    # get userid
    user_id = rand_tweet_df['user_id'].iloc[i]
    # lookup index of userid in rand_user_timelines
    line_id = rand_user_timelines[rand_user_timelines['user_id'] == user_id].index[0]
    # add corresponding feature vector to bert_vecs
    spacy_vecs.append(spacy_features[line_id])
# missing: adjacency_list
with open("data_intermediary-small/adjacency_list_76.pickle", "rb") as f:
    adj_list = pickle.load(f)
new_adf_list = []
for i in range(len(adj_list)):
    tweet_1 = rand_tweet_df[rand_tweet_df['tweet_id'] == adj_list[i][0]]
    tweet_2 = rand_tweet_df[rand_tweet_df['tweet_id'] == adj_list[i][1]]
    if len(tweet_1) > 0 and len(tweet_2) > 0:
        new_adf_list.append(adj_list[i])
    if len(tweet_1) == 0 and len(tweet_2) > 0:
        print("Why do we only find tweet2?", i)
    if len(tweet_2) == 0 and len(tweet_1) > 0:
        print("Why do we only find tweet1?", i)
import format_data
spacy_vecs_np = np.array(spacy_vecs)
bert_vecs_np = np.array(bert_vecs)
format_data.reorganize_dump_all(rand_tweet_df, claim_df, new_adf_list, spacy_vecs_np, bert_vecs_np, [], [], "correct_bert_spacy")

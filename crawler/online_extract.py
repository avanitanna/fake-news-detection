from datetime import datetime
import spacy
import numpy as np
import pickle
from clip_client import Client
import scipy.sparse
"""

Helper code to generate 10-dimensional user profile feature based on crawled user object using Twitter Developer API

"""
def extract_profile_features(user_df, tweet_df):
    """ Extract 10-dimensional user profile features for all users. """
    feature = np.zeros([len(tweet_df), 10], dtype=np.float32)
    id_counter = 0
    est_date = datetime.fromisoformat('2006-03-21') #CHANGE THIS!
    for i in range(len(tweet_df)):
        user_id = tweet_df['user_id'].iloc[i]
        users = user_df[user_df['user_id'] == user_id]['user_obj']
        user = users.iloc[0]
        #user = user_df['user_obj'][i]
        # 1) Verified?, 2) Enable geo-spatial positioning, 3) Followers count, 4) Friends count
        vector = [int(user.verified), int(user.geo_enabled), user.followers_count, user.friends_count]
        # 5) Status count, 6) Favorite count, 7) Number of lists
        vector += [user.statuses_count, user.favourites_count, user.listed_count]

        # 8) Created time (No. of months since Twitter established)
        user_date = user.created_at #datetime.strptime(user_df['created_at'][i], '%a %b %d %H:%M:%S +0000 %Y')
        month_diff = (user_date.year - est_date.year) * 12 + user_date.month - est_date.month
        vector += [month_diff]

        # 9) Number of words in the description, 10) Number of words in the screen name
        vector += [len(user.name.split()), len(user.description.split())]

        feature[id_counter, :] = np.reshape(vector, (1, 10))
        id_counter += 1
    return feature

def extract_spacy_features(tweet_df):
    """ Extract word embeddings using spacy. """
    nlp = spacy.load('en_core_web_lg')
    spacy_vecs = []
    for t in tweet_df['text']:
        spacy_vecs.append(nlp(t).vector)
    return spacy_vecs

def extract_bert_features(tweet_df):
    """ Extract word embeddings using bert/clip. """
    c = Client('grpc://0.0.0.0:51000')
    bert_vecs = c.encode(tweet_df['text'])
    return bert_vecs

def build_content_features(spacy_vec, profile_vec):
    """ Create the content feature vectors. """
    return np.column_stack([spacy_vec, profile_vec])

def save_features(vec,index,name):
    scipy.sparse.save_npz(f"data/new_{name}_feature_{index}.npz",vec) 

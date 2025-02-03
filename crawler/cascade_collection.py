import tweepy
from mumin import MuminDataset
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import time
from online_extract import *
from threading import *
import copy
from format_data import *

class CascadeFeatureDump:

    def __init__(self):

        self.dataset = MuminDataset(twitter_bearer_token="")
        self.dataset.compile()

        self.claim_df = self.dataset.nodes['claim']

        self.tweet_df = self.dataset.nodes['tweet']
        #self.tweet_df = self.tweet_df.dropna()
        self.tweet_df['is_retweet'] = 0
        self.tweet_df['graph_label'] = -1
        self.tweet_df['user_id'] = 0
        self.tweet_df = self.tweet_df[["tweet_id", "text", "created_at",
            "is_retweet","graph_label","user_id"]]
        self.tweet_df = self.tweet_df.astype({'tweet_id': np.uint64,
            'text': object, 'created_at': 'datetime64[ns]',
            'is_retweet': np.int64, 'graph_label': np.int64,
            'user_id': np.uint64})
        self.tweet_df['graph_label'] = np.nan

        self.tweet_claim = self.dataset.rels[('tweet','discusses','claim')]  #src and tgt columns

        self.col_names = ["user_id", "user_obj"]
        self.user_df = pd.DataFrame(columns = self.col_names)

        # assign consumer and access key/secret values
        self.consumer_key = ""
        self.consumer_secret = ""
        self.access_token = ""
        self.access_token_secret = ""

        # authorization of consumer key and consumer secret
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        # set access to user's access key and access secret 
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        # calling the api 
        self.api = tweepy.API(self.auth,wait_on_rate_limit=True)

        self.tweet_retweet_relations = []
        self.tweet_id_set = set(self.tweet_df['tweet_id'])
        self.user_id_set = set()

        # creating thread instance where count = 1
        self.sema_obj = Semaphore(1) 
        self.tweet_last =  len(self.tweet_df)
        self.user_last =  0
        self.tweet_retweet_last = 0

    def preprocess(self):
        ## preprocessing - graph labels of existing tweets
        n = len(self.tweet_df)

        for i in range(len(self.tweet_claim)):
            if self.tweet_claim['src'][i] >= len(self.tweet_df):
                continue
            print("index ",str(i)," label ",str(self.tweet_claim['tgt'][i]))
            self.tweet_df['graph_label'][self.tweet_claim['src'][i]] = self.tweet_claim['tgt'][i]

        ## preprocessing - get status for existing tweets and fill in the 'is_retweet' values
        for i in range(n):
            print("tweet "+str(i)+", "+str(self.tweet_df['tweet_id'][i]))
            try:
                status = self.api.get_status(self.tweet_df['tweet_id'][i], tweet_mode="extended")
                # ## preprocessing - get user objects for existing tweets - add user object and id
                udf = pd.DataFrame([{'user_id':status.user.id,
                            'user_obj':status.user}]).astype({'user_id':np.uint64,
                            'user_obj':object})
                self.user_df = pd.concat([self.user_df,udf],ignore_index=True)
                self.user_id_set.add(status.user.id)
                self.user_last+=1
                self.tweet_df['user_id'][i] = status.user.id #add user id to the tweet df
            except:
                print("forbidden access for tweet "+str(i)+", "+str(self.tweet_df['tweet_id'][i]))
                self.tweet_df['is_retweet'][i] = np.nan #tweet access forbidden 
                continue
            try:
                print(status.retweeted_status)
                self.tweet_df['is_retweet'][i] = 1
            except:  # Not a Retweet
                #print(status.full_text)
                pass
        self.tweet_df = self.tweet_df.dropna().reset_index(drop=True)

        """
        ## fix graph labels
        labels = list(set(self.tweet_df['graph_label']))
        d = {}
        for i in range(len(labels)):
            d[labels[i]] = i
        for i in range(len(self.tweet_df)):
            graph_label = self.tweet_df['graph_label'][i]
            self.tweet_df['graph_label'][i] = d[graph_label]

        #### one off dump for graph_labels : consisting of 0 or 1 values - misinformation or factual
        graph_labels = np.zeros(len(labels), dtype=np.int64)
        for i in range(len(labels)):
            if self.claim_df['label'][int(labels[i])] == "factual":
                graph_labels[i] = 1
        with open(f"data/graph_labels.npy", "wb") as f:
            np.save(f,graph_labels)
        """

    def collect_cascade(self):
        # call dump func here - in a separate thread 
        i = 0
        while(True):
            print("tweet "+str(i)+", "+str(self.tweet_df['tweet_id'][i]))
            try:
                retweets_list = self.api.get_retweets(self.tweet_df['tweet_id'][i])
            except:
                # -1 means status is forbidden, -2 means retweets is forbidden
                self.tweet_df['is_retweet'][i] = -2
                i += 1
                continue
            for retweet in retweets_list:
                if retweet.id in self.tweet_id_set:
                    continue
                else:
                    self.tweet_id_set.add(retweet.id)
                    df = pd.DataFrame([{'tweet_id':retweet.id, 'text':retweet.text,
                        'created_at':retweet.created_at, 'is_retweet':1, 'graph_label':self.tweet_df['graph_label'][i], 
                        'user_id':retweet.author.id}]).astype({"tweet_id": np.uint64,
                        "text": object, "created_at": 'datetime64[ns]', "is_retweet":np.int64,
                        "graph_label":np.int64, "user_id":np.uint64})
                    self.tweet_df = pd.concat([self.tweet_df,df],ignore_index=True)

                    if retweet.author.id not in self.user_id_set:
                        udf = pd.DataFrame([{'user_id':retweet.author.id,
                            'user_obj':retweet.author}]).astype({'user_id':np.uint64,
                            'user_obj':object})
                        self.user_df = pd.concat([self.user_df,udf],ignore_index=True)
                        self.user_id_set.add(retweet.author.id)
                self.tweet_retweet_relations.append([self.tweet_df['tweet_id'][i],retweet.id])

                # call acquire, set last values and release
                self.sema_obj.acquire()
                self.tweet_last = len(self.tweet_df)
                self.user_last = len(self.user_df)
                self.tweet_retweet_last = len(self.tweet_retweet_relations)
                self.sema_obj.release()

            i+=1

    def process_and_dump(self):

        start_tweet = 0
        start_user = 0
        bert_vecs = np.empty(shape=[0,768])
        spacy_vecs = np.empty(shape=[0,300])
        profile_vecs = np.empty(shape=[0,10])
        content_vecs = np.empty(shape=[0,310])
        iter_idx = 0
        while(True):

            #sleep for 2 hours/7200 s
            print("process_and_dump sleeping now...")
            time.sleep(7200) # 2h
            print("process_and_dump awake now...")
            ## use semaphore to acquire last position of the tweet and user df as well as tweet retweet relations
            ## As graph labels is a hashmap, use deepcopy
            self.sema_obj.acquire()
            end_tweet = self.tweet_last
            end_user = self.user_last
            end_tweet_retweet = self.tweet_retweet_last
            #user_id_set_copy = copy.deepcopy(self.user_id_set)
            #tweet_id_set_copy = copy.deepcopy(self.tweet_id_set)
            self.sema_obj.release()

            ## dump data - tweet, user, graph and relations
            with open(f"data_intermediary/tweet_df_{iter_idx}.pickle","wb") as f:
                pickle.dump(self.tweet_df.iloc[:end_tweet,:],f)

            # with open(f"data/tweet_ids_{iter_idx}.pickle","wb") as f:
            #     pickle.dump(list(tweet_id_set_copy),f)

            # with open(f"data/user_ids_{iter_idx}.pickle","wb") as f:
            #     pickle.dump(list(user_id_set_copy),f)

            with open(f"data_intermediary/user_df_{iter_idx}.pickle","wb") as f:
                pickle.dump(self.user_df.iloc[:end_user,:],f)

            with open(f"data_intermediary/adjacency_list_{iter_idx}.pickle","wb") as f:
                pickle.dump(self.tweet_retweet_relations[:end_tweet_retweet],f)

            ## perform feature extraction (incrementally)
            if start_tweet < end_tweet:
                bert_vecs = np.concatenate((bert_vecs,extract_bert_features(self.tweet_df.iloc[start_tweet:end_tweet,:])))
                spacy_vecs = np.concatenate((spacy_vecs,extract_spacy_features(self.tweet_df.iloc[start_tweet:end_tweet,:])))
                profile_vecs = np.concatenate((profile_vecs,extract_profile_features(self.user_df.iloc[:end_user,:],
                    self.tweet_df.iloc[start_tweet:end_tweet,:])))
                content_vecs = np.array(build_content_features(spacy_vecs,profile_vecs))

            # CALL format data 
            reorganize_dump_all(self.tweet_df.iloc[:end_tweet,:],self.claim_df,
                self.tweet_retweet_relations[:end_tweet_retweet],spacy_vecs,bert_vecs,profile_vecs,content_vecs,iter_idx)

            # reset start variable 
            start_tweet = end_tweet
            start_user = end_user
            iter_idx+=1

def main():
    obj = CascadeFeatureDump()
    obj.preprocess()
    t1 = Thread(target = obj.collect_cascade)
    t1.start()
    obj.process_and_dump()

if __name__ == "__main__":
    main()

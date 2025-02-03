import pickle
import pandas
import tweepy
from mumin import MuminDataset


def init():
    auth = tweepy.OAuth1UserHandler(
    consumer_key='', 
    consumer_secret='', 
    access_token='', 
    access_token_secret=''
    )
    api = tweepy.API(auth,wait_on_rate_limit=True)

    a = open("/home/raavi/fake-news/fake-news-detection/mumin-raw/adjacency_list_76.pickle","rb")
    a_df = pickle.load(a)

    # {root : [nodes that come from root]}
    adj_dict = {}
    for vector in a_df:
        if vector[0] not in adj_dict:
            adj_dict[vector[0]] = [vector[1]]
        else:
            adj_dict[vector[0]].append(vector[1])

    u = open("/home/raavi/fake-news/fake-news-detection/mumin-raw/user_df_76.pickle","rb")
    u_df = pickle.load(u)

    # {user id : user obj}
    user_dict = {}
    for x in range(len(u_df['user_id'])):
        #print(u_df['user_id'][x])
        user_dict[u_df['user_id'][x]] = u_df['user_obj'][x]

    t = open("/home/raavi/fake-news/fake-news-detection/mumin-raw/tweet_df_76.pickle","rb")
    t_df = pickle.load(t)

    # {tweet id : user id}
    tweet_dict = {}
    for x in range(len(t_df['tweet_id'])):
        tweet_dict[t_df['tweet_id'][x]] = t_df['user_id'][x]

    dataset = MuminDataset(twitter_bearer_token='',size='small')
    dataset.compile()
    follow_df = dataset.rels[('user', 'follows', 'user')]
    follow_dict = {}
    for i in range(len(follow_df['src'])):
        if int(follow_df['src'][i]) not in follow_dict:
            follow_dict[int(follow_df['src'][i])] = [int(follow_df['tgt'][i])]
        else:
            follow_dict[int(follow_df['src'][i])].append(int(follow_df['tgt'][i]))

    return api, adj_dict, user_dict, tweet_dict, follow_dict

API,ADJ,USER,TWEET, FOLLOWING = init()


def print_adjacencies(adj_dict):
    for entry in adj_dict:
        print(f"Base: {entry}, Leaves: {adj_dict[entry]}\n")


# creates ordered cascade by date
# dataset was already ordered by timestamp
def create_ordered_cascade(id):
    #print(f"Root is {id}")
    o = []
    for leaf in ADJ[id]:
        try:
            o.append(leaf)
        except:
            print(f"Leaf ID: {leaf}, NO TWEET FOUND")
    o.append(id)
    o.reverse()
    return o


def create_user_id_list(tweet_list):
    user_id = []
    for i in tweet_list:
        try:
            #status = API.get_status(i)
            u_id = TWEET[i]
            user_id.append(u_id)
        except:
            continue
    return user_id
    

def get_followers(user_id):
    try:
        user = USER[user_id]
        return user._json['followers_count']
    except:
        return 0


follow_rel = {'follower' : [], 'followee' : []}
# TODO: store in pandas dataframe, two columns, [follower, followee]
def is_following(source_id,target_id):  
    following = False
    # check mumin for follow rel.
    #friendship = API.get_friendship(source_id=source_id, target_id=target_id)
    #following = friendship[1].following
    try:
        if target_id in FOLLOWING[source_id]:
            following = True
    except:
        print(f">>> No ID Found: {source_id}")
        try:
            friendship = API.get_friendship(source_id=source_id, target_id=target_id)
            following = friendship[1].following
            if following:
                if source_id not in FOLLOWING.keys():
                    FOLLOWING[source_id] = [target_id]
                else:
                    FOLLOWING[source_id].append(target_id)
        except:
            print(f"ID: {source_id} | No Follow Relation")
    if following:
        follow_rel["follower"].append(source_id)
        follow_rel["followee"].append(target_id)
    return following


def top_followers(user_id_list):
    top_followers = 0
    top_id = 0
    for i in user_id_list:
        cur_foll = get_followers(i)
        #print(f"User: {i}, {cur_foll} Followers")
        if cur_foll > top_followers:
            top_followers = cur_foll
            top_id = i
    return top_id


def adjacency_creator(adjacency_dict):
    new_adj = []
    for i in adjacency_dict:
        o = create_ordered_cascade(i)
        for x in range(len(o)):
            if x == 0:
                continue
            else:
                curr_tw_id = o[x]
                curr_user_id = TWEET[curr_tw_id]
                checklist = o[0:x]
                checklist.reverse()
                user_checklist = {}
                has_follow = False
                for cl_tw_id in checklist:
                    y_user_id = TWEET[cl_tw_id]
                    user_checklist[y_user_id] = cl_tw_id
                    if is_following(curr_user_id,y_user_id):
                        new_edge = [curr_tw_id,cl_tw_id]
                        new_adj.append(new_edge)
                        has_follow = True
                        break
                if has_follow != True:
                    top_id = top_followers(user_checklist)
                    new_edge = [curr_tw_id,user_checklist[top_id]]
                    new_adj.append(new_edge)
                    user_checklist.clear()
    return new_adj


#print_adjacencies(ADJ)
fake_ADJ = { 1337327032418627585: [1338377084478099458, 1337789880936574976, 1337787943868649474, 1337539333784539138, 1337518981205331976, 1337508780171517954, 1337500260428636160, 1337482634390335491, 1337481160256073741, 1337471703027503105, 1337466205377060865, 1337461808341413889, 1337458822403776512],
            1337292732600111104 : [1337433971412627456, 1337299596255633410, 1337296997708128261, 1337294076132143105, 1337293784623755264, 1337293640532717569],
            1337473136225378309 :  [1337682562857246725, 1337656518959898624, 1337656022828281858, 1337487339929661442, 1337475607467274242, 1337475240067272706, 1337474980926337025, 1337474211758149634]}


nadj = adjacency_creator(fake_ADJ)
print(nadj)
nadj_df = pandas.DataFrame(nadj)
nadj_df.to_pickle("/home/raavi/fake-news/fake-news-detection/adjacency_list_h.pickle")
following_df = pandas.DataFrame(data=follow_rel)
following_df.to_pickle("/home/raavi/fake-news/fake-news-detection/follow_df_h.pickle")


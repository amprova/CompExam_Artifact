
import pandas as pd
import numpy as np
from scipy import sparse as sps
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

import lenskit
import lenskit.crossfold as xf
from  lenskit.crossfold import TTPair


from lenskit.algorithms import als, basic, item_knn, user_knn
from lenskit.algorithms.basic import Fallback
from lenskit.algorithms.als import BiasedMF, ImplicitMF
from lenskit.algorithms.implicit import BPR

from lenskit.batch import MultiEval
from lenskit.crossfold import partition_users, SampleN
from lenskit.crossfold import sample_users, SampleN
from lenskit import batch, topn, util
from TF_IDF_final import tf_idf
from LDA_Final import LDAKL
'''
saved = open("pickle/game_reviews.pickle","rb")
game_reviews = pickle.load(saved)

user_game = open("pickle/user_games.pickle","rb")
user_games = pickle.load(user_game)

user_games = user_games.rename(columns={'user_id': 'user', 'item_id': 'item'})
user_games_list = user_games[['item', 'user']]


reviews = game_reviews[['item_id', 'user_id','review']]
reviews = reviews.rename(columns={'user_id': 'user', 'item_id': 'item'})

rev_item = set(reviews['item'])
user_item = set(user_games['item'])
item_butNot_rev = user_item.intersection(rev_item) ## items that have reviews

user_item_rev = user_games_list[user_games_list['item'].isin(list(item_butNot_rev))] 

result = pd.merge(user_item_rev, reviews, how = 'outer', on=['item', 'user'])



def groupby_count(df, group, count):
    game_count = pd.DataFrame()
    game_count['count'] = df.groupby(group)[count].count()
    return game_count


def prune(df, condition):     ## returns a dataframe that meet the given condition
    user_n = df.loc[df['count'] < condition ]
    return user_n


game_count = groupby_count(result, 'user', 'item')


user_5 = prune(game_count, 5)


user_less_5 = user_5.index
user_less_5



pruned_data_5 = result.set_index('user').drop(user_less_5)
pruned_data_5.reset_index(inplace = True)

#pairs_user = list(partition_users(pruned_data_5, 5, xf.SampleN(1)))
pairs_user = list(sample_users(pruned_data_5, 5, 12000, xf.SampleN(1) ))
pickle_out = open("sample_user.pickle","wb")
pickle.dump(pairs_user, pickle_out)
pickle_out.close()

truth = pd.concat((p.test for p in pairs_user))
#truth.to_csv(r'results/steam/pruned_5.csv')

'''
pickle_in = open("aspect_sample_user.pickle","rb")
pairs_user = pickle.load(pickle_in)

def algo_eval(path, algo, dataset):
    evaluation = batch.MultiEval(path=path, predict=False, recommend=100, n_jobs = 6)
    evaluation.add_algorithms(algos=algo)
    evaluation.add_datasets(data=dataset)
    evaluation.run()


#algo_ii = item_knn.ItemItem(20, center=False, aggregate='sum')
#algo_uu = user_knn.UserUser(30, center=False, aggregate='sum')
#algo_pop = basic.Popular()
#algo_mf = ImplicitMF(40)
#algo_bpr = BPR()
algo_tf_idf = tf_idf()
algo_LDA = LDAKL()

algo_eval('results/az/tf_LDA', [algo_LDA, algo_tf_idf], pairs_user)



import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
from scipy import sparse
from sklearn.preprocessing import normalize
import logging
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from lenskit import util
from sklearn.decomposition import LatentDirichletAllocation
logging.basicConfig(filename='LDA.log',filemode='a',level=logging.INFO)
_logger = logging.getLogger(__name__)


class ReviewLDAKL:
    
    """
    class ReviewLDAKL(tokenize = True, lower = True, stop_words_removal = True, stemmed = True, NUM_TOPICS=20)
    
    ReviewLDA is an item-item similarity calculation algorithm based on topic distribution. 
    Item features are generated based on topic distribution from user reviews. Latent Dirichlet Allocation (LDA) is used to model topics from reviews. Before applying LDA reviews can be processed (tokenized, lowercased, stopwords removed, stemmed). To find the similarity between item topic distribution, Jensen-Shannon divergence is calculated.

    Args:
        tokenize: bool, (default True) whether to tokenize the user reviews
             Nltk.tokenize.regexp is used to tokenize the reviews.
        lower: bool, (default True) whether to lowercase all the tokens of reviews
        stop_words_removal: bool, (default True) whether to remove the stop words
             English stopwords from nltk.corpus are used to identify stopwords. 
        stemmed: bool, (default True) whether to convert the tokens into stemmed form
             Nltk.stem.PorterStemmer is used to stem the tokens.
        NUM_TOPICS: int, (default 20)
             Number of topics to calculate the topic distribution of items.

    """
    
    similarity_matrix = None
    review_data = None
    item_data = None
    timer = None
    user_index = None
    LDA_matrix = None
    item2index = None
    #NUM_TOPICS = 20
    
    
    def __init__(self, tokenize = True, lower = True, stop_words_remove = True, stemmed = True, NUM_TOPICS=20):
        
        self.tokenize = tokenize
        self.lower = lower
        self.stop_words_remove = stop_words_remove
        self.stemmed = stemmed
        self.NUM_TOPICS = NUM_TOPICS
        
    def process(self, content):
        
        if self.tokenize is True:
            processed = tokenizer.tokenize(content)
        else:
            processed = content.split()
        if self.lower is True:
            processed = [token.lower() for token in processed]
        if self.stop_words_remove is True:
            processed = [token for token in processed if token not in stopwords.words('english')]
        if self.stemmed is True:
            processed = [ps.stem(token) for token in processed]
        
        return processed
     
    def LDA(self, data_table, col_name):
        
        lda = LatentDirichletAllocation(n_components=self.NUM_TOPICS, random_state=0)
        vect = CountVectorizer(tokenizer=self.process)
        corpus = data_table[col_name].tolist()
        BOW = vect.fit_transform(corpus)
        LDA_mat = lda.fit_transform(BOW)
        LDA_MAT = sparse.csr_matrix(LDA_mat)
        return LDA_MAT.todense()

    def jensen_shannon(self, user_item, target_items):
        p = user_item.T # take transpose
        q = target_items.T # transpose matrix
        m = 0.5*(p + q)
        dist = np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))
        similarity_score = 1 - dist
        return similarity_score

    def get_user_item(self, userID):
        
        item_list = self.user_index.loc[userID]
        if isinstance(item_list, str):
            item_list = pd.Series(item_list).rename("item")
        temp_df = item_list.to_frame()
        temp_df = temp_df.reset_index()
        return temp_df
    
    def itemid_2_index(self, item):
        try:
            return self.item2index.get_loc(item)
        except KeyError:
            pass
    
    def user_item_topic_dist(self, userID):
        temp_df = self.get_user_item(userID)
        items = temp_df['item'].unique()
        itemIX = [self.itemid_2_index(x) for x in items if self.itemid_2_index(x) is not None]
        UI_topic_dist = self.LDA_matrix[itemIX, :].sum(axis = 0)
        return UI_topic_dist
        
    def sim_score(self, UI_topic_dist,target):
        
        target_idx=pd.Series()
        for item in target:
            ix = self.itemid_2_index(item)
            if ix is not None:
                target_idx.at[item] = ix
        target_mat = self.LDA_matrix[target_idx, :]
        sims = self.jensen_shannon(UI_topic_dist, target_mat)
        item_sim = pd.Series(sims, name='rev_sim', index=target_idx.index)
    
        return item_sim
       

    def fit(self, pruned_data):
        
        """
        Train the model using the specified reviews data.
        Args:
            reviews(pandas.dataframe): the reviews data
        Returns:
            The algorithmic object
        """
        
        self.timer = util.Stopwatch()
        self.review_data = pruned_data
        only_rev = pruned_data.dropna()
        
        item_rev = pd.DataFrame({'review': only_rev.groupby(['item']).review.apply(lambda x:' '.join(x))})
        item_rev.reset_index(inplace=True)
        self.item_data = item_rev
        self.LDA_matrix = self.LDA(self.item_data, 'review')
        self.user_index = self.review_data.set_index('user')['item']
        self.item2index = pd.Index(self.item_data.item.unique(), name='item')
        _logger.info('[%s] fitting LDA model', self.timer)
        
        return self
        
        
    def predict_for_user(self, userID, itemList, ratings = None):
        
        """
        Compute prediction for users and items.

        Args:
            user: the userID
            items (array-like): the items to predict
            ratings: None
        Returns:
            scores for the items, indexed by item id
        Return type: pandas.Series

        """
        if userID in self.user_index.index:
            UI_dist = self.user_item_topic_dist(userID)
            final_score = self.sim_score(UI_dist, itemList)

            _logger.info('[%s] Predicting items for UserID %s', self.timer, userID)
            final_score.index.name = 'item'
            return final_score
        else:
            return pd.Series(np.nan, index=itemList)
        
    def __str__(self):
        return 'LDA'
    
        
            
         
import numpy as np
import pandas as pd
from modules import util


class Vectorizer:
    """
    This class handles a sequence of tokens' sequences and their transformation to numbers.
    
    """
    def __init__(self):
        self.tokens = []
        self.freq_dist = dict()
           
    def fit(self, tokens):
        """
        Construct the vocabulary, the frequency distribution, and a ranked frequencey from tokens.
        
        Parameters
        ----------
        tokens: 
            An iterable containing iterables of tokens or a generator to generate an iterable of tokens at every iteration.
              
        Returns
        -------
        type
           The instance.
        
        """
        assert util.is_iter(tokens), "Parameter tokens must be an iterable of iterables."
        self.tokens = tokens
        self.__vocabulary__()
        return self

    def transform(self, tokens, keep_n_most_frequent=None, nonkeyvalue=0, ):
        """
        Transform a sequence of tokens' sequences to numbers using ranked frequency. 
        
        Parameter
        ---------
        tokens: iterable of iterables or a generator to generate a sequence of tokens.
           The data to be transformed.
        
        nonkeyvalue: int
            An integer to transform tokens not existed in vocabulary.
        
        keep_n_most_frequent: int or None
            Tokens with ranked frequency less than this number will take the nonkeyvalue. Default: None. Keep all ranks.
            
        Returns
        -------
        list
            A listcontaining lists of the transformed values.
        
        """
        ranker = pd.Series(self.freq_rank).iloc[:keep_n_most_frequent]
        return pd.Series(tokens).apply(lambda tl: [ranker.get(t, nonkeyvalue) for t in tl]).tolist()
    
    @property
    def shape(self):
        return np.shape(self.tokens) or type(self.tokens)
    
    @property
    def n_vocabulary(self):
        return len(self.freq_dist)
    
    @property
    def vocabulary(self):
        return set(self.freq_dist.keys())
    
    @property
    def freq_dist_descending(self):
        return util.sort_dict(self.freq_dist, reverse=True)
    
    @property
    def freq_dist_ascending(self):
        return util.sort_dict(self.freq_dist, reverse=False)
    
    @property
    def vocabulary_freq_ranked(self):
        return list(self.freq_dist_descending)
    
    @property
    def freq_rank(self):
        keys = self.freq_dist_descending.keys()
        values = list(range(1, len(keys)+1))
        return util.OrderedDict(zip(keys, values))
      
    def __vocabulary__(self):
        gen = (t for t in self.tokens)
        while True:
            try:
                curr = gen.__next__()
                for t in curr:
                    self.freq_dist[t] = self.freq_dist.get(t, 0) + 1 
            except StopIteration:
                return None 
        
     
    def __repr__(self):
        return f"<TokenTransformer(sequences={self.shape}, n_vocabulary={self.n_vocabulary})>"
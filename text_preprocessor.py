# -*- coding: utf-8 -*-
"""
"""

import nltk
import pandas as pd


class TextsPreprocessor:
    """
    This class preprocesses a sequence of texts. This preprocessing is suggested for formal texts like
    articles consisted of several sentences where the sentences' ends might be useful to be indicated
    in the corresponding vector. The class provides a transformation pipeline on a specific order that
    can be changed or certain transforms might be deactivated. The final result provided by the
    'transform' method can be a sequence of tokens' sequences or a sequence of transformed texts.

    The transformations supported are presented below.  
    # sentence tokenize
    # word tokenize
    # remove uppercase words
    # clean tokens starting with upper letter when not the first one in a sentence.
    # clean non-alphanumeric tokens except the last one in a sentence.
    # lower tokens
    # remove stopwords
    # lemmatize
    # stem

    """
    transformation_order = ['clean_upper',
                            'clean_title',
                            'clean_non_alpha',
                            'lower',
                            'remove_stopwords',
                            'replace_tokens',
                            'lemmatize',
                            'stem',
                            'flatten',
                            'reconstruct']

    transformation_selection = dict(zip(transformation_order, [True] * len(transformation_order)))
    non_alpha_exceptions = []
    replacer = dict()
    extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e']

    def __init__(self, stopwords=None, lemmatizer=None, stemmer=None, word_tokenize=None, sent_tokenize=None):
        self.sw = stopwords or self.default_stopwords
        self.lm = lemmatizer or self.default_lemmatizer
        self.sm = stemmer or self.default_stemmer
        self.word_tokenize = word_tokenize or self.default_word_tokenize
        self.sent_tokenize = sent_tokenize or self.default_sent_tokenize

        assert type(self.sw) is list and all(type(s) is str for s in self.sw), "Parameter 'stopwords' must be a list of strings."
        assert hasattr(self.lm, 'lemmatize'), "Lemmatizer must support a 'lemmatize' method."
        assert type(self.lm.lemmatize('queen')) is str, "Method lemmatize must return a string."
        assert hasattr(self.sm, 'stem'), "Stemmer must support a 'stem' method."
        assert type(self.sm.stem('queen')) is str, "Method stem must return a string."
        assert type(self.word_tokenize) is type(lambda: True), "Parameter 'word_tokenize' must be a function."
        assert type(self.sent_tokenize) is type(lambda: True), "Parameter 'sent_tokenize' must be a function."
        
        try:
            t = self.word_tokenize("blah blah")
            assert all(type(e) is str for e in t)
        except:
            raise Exception("Function word_tokenize must return a list of strings")
            
        try:
            t = self.sent_tokenize("blah blah. More blah blah!")
            assert all(type(e) is str for e in t)
        except:
            raise Exception("Function sent_tokenize must return a list of strings")
        assert type(self.sent_tokenize) is type(lambda: True), "Parameter 'sent_tokenize' must be a function."
        
        self.texts = None
        self.sentences = None
        self.tokens = None
        self.current = None
        
    @property   
    def default_stopwords(self):
        return nltk.corpus.stopwords.words('english')
    
    @property
    def default_lemmatizer(self):
        return nltk.WordNetLemmatizer()
    
    @property
    def default_stemmer(self):
        return nltk.PorterStemmer()
    
    @property
    def default_word_tokenize(self):
        return nltk.word_tokenize
    
    @property
    def default_sent_tokenize(self):      
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentence_tokenizer._params.abbrev_types.update(self.extra_abbreviations)
        return nltk.sent_tokenize

    def fit(self, texts):
        """
        Fit a sequence of strings to be tokenized first into sentences and then into words. The method will create a  Series where the value of every row consists of nested lists of tokenized sentences.
    
        Example
        -------
        >>> tp = TextsPreprocessor()
        >>> texts = ["I am the first one. You are the second!",
                    "Go forth! Go now...",
                    "Come on, I need you here. Now? \nYes, now!"]
        >>> tp.fit(texts)
        >>> tp.tokens
        out:
            0             [[I, am, the, first, one, .], [You, are, the, second, !]]
            1                                      [[Go, forth, !], [Go, now, ...]]
            2    [[Come, on, ,, I, need, you, here, .], [Now, ?], [Yes, ,, now, !]]
                dtype: object
        """
        assert all(type(t) is str for t in texts), "Parameter 'texts' must be a sequence of strings."
        self.texts = pd.Series(texts)
        self.sentences = self.texts.apply(self.sent_tokenize)
        self.tokens = self.sentences.apply(self.word_tokenize_sentlist, word_tokenize=self.word_tokenize)
        return self

    def transform(self, verbose=1, **kwargs):
        """
        Apply the transformations on the tokens. Deactivate or activate transformations by setting their
        name as False, True, respectively.

        Parameters
        ----------
        verbose: int
            Set 0 for no verbosity. Default: 1.
        *kwargs: arguments
            Keys of the transformation_selection dictionary that can be set to True or False to respectively activate
            or deactivate a transformation.

        Returns
        -------
        pd.Series
            The type of the Series values can be list of tokens or texts dependent on the final transformation.
        """
        self.current = self.tokens.copy()
        transformation_selection = self.transformation_selection.copy()
        for kw in kwargs:
            if kw in transformation_selection:
                transformation_selection[kw] = kwargs[kw]
        for i, trans in enumerate(self.transformation_order, 1):
            if verbose:
                print(f"{i}/{len(self.transformation_order)}. {trans}...", end='')
            if  transformation_selection[trans]:
                self.current = self.current.apply(eval(f"self.{trans}", ), 
                                                  stopwords=self.sw, 
                                                  lemmatizer=self.lm, 
                                                  stemmer=self.sm,
                                                  exceptions=self.non_alpha_exceptions,
                                                  replacer=self.replacer)
            if verbose:
                print("Completed." if transformation_selection[trans] else "Skipped.")
        return self.current
                
    def __repr__(self):
        return "<TextsPreprocessor>"
    
    @staticmethod
    def word_tokenize_sentlist(sentlist, **kwargs):
        return [kwargs['word_tokenize'](sent) for sent in sentlist]   
    
    @staticmethod   
    def clean_upper(tokenlists, **kwargs):
        return [[t for t in tl if not t.isupper()] for tl in tokenlists]
                    
    @staticmethod
    def clean_title(tokenlists, **kwargs):
        return [[t for t in tl[1:] if not t.istitle()] for tl in tokenlists]
    
    @staticmethod
    def clean_non_alpha(tokenlists, **kwargs):
        #return [[t for t in (tl[:-1] if len(tl) > 1 else tl) if t.isalpha()] + ([tl[-1]] if len(tl) > 1 else []) for tl in tokenlists] 
        return [[t for t in tl if t in kwargs['exceptions'] or t.isalpha()] for tl in tokenlists]
    
    @staticmethod
    def lower(tokenlists, **kwargs):
        return [[t.lower() for t in tokenlist] for tokenlist in tokenlists]
    
    @staticmethod
    def remove_stopwords(tokenlists, **kwargs):
        return [[t for t in tokenlist if t not in kwargs['stopwords']] for tokenlist in tokenlists]
    
    @staticmethod
    def replace_tokens(tokenlists, **kwargs):
        replacer = kwargs['replacer']
        return [[replacer[t] if t in replacer else t for t in tokenlist] for tokenlist in tokenlists]
    
    @staticmethod
    def lemmatize(tokenlists, **kwargs):
        return [[kwargs['lemmatizer'].lemmatize(t) for t in tokenlist] for tokenlist in tokenlists]
    
    @staticmethod
    def stem(lemmalists, **kwargs):
        return [[kwargs['stemmer'].stem(lemma) for lemma in lemmalist] for lemmalist in lemmalists]
    
    @staticmethod
    def flatten(tokenlists, **kwargs):
        flattened = []
        for tl in tokenlists:
            flattened.extend(tl)
        return flattened
    
    @staticmethod
    def reconstruct(tokenlist, **kwargs):
        return " ".join(tokenlist)
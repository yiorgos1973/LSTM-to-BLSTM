from nltk.corpus import stopwords as SW
from modules import util

class Stopwords:
    """
    This is a class to handle stopwords. 
    
     
    Attributes
    ----------
    stopwords: list
        A list of strings containing the current stopwords. Default: The nltk stopwords.
    len: int
       The current number of stopwords.
    default: list
        The list of default stopwords. Equals to nltk english stopwords.
       
       
    Methods
    -------
    filter():
        An iteration over the current stopwords with user interaction for identifying the ones to keep.        
    add(*stopwords):
       Add one or more stopwords.   
    remove(*stopwords):
       Remove one or more stopwords.    
    default_different():
        Return the stopwords not contained into the nltk stopwords.    
    default_missing():
        Return the nltk stopwords not contained into the current stopwords.    
    save():
       Save current stopwords.    
    load(fp):
        Replace the current stopwords with a loaded pickle file.
        
    """       
 
      
    def __init__(self):
        self.stopwords = self.default
    
    @property
    def len(self):
        return len(self.stopwords)
    
    @property
    def default(self):
        return list(SW.words('english'))
    
    def filter(self):
        """Filter current stopwords with user interaction.""" 
        old_len = self.len
        for i, sw in enumerate(self.stopwords.copy(), 1):
            print(f"{i}/{old_len}. ", end='')
            if not self.__is_stopword__(sw):
                self.remove(sw)
        print(f"\033[1m'Filtering completed.\033[0m \n")
        return self
    
    def add(self, *stopwords):
        """Add one or more stopwords."""
        assert all(type(s) is str for s in stopwords), "All stopwords must be strings."
        add = [s for s in stopwords if not s in self.stopwords]
        print(f"{len(add)} stopwords added.")
        self.stopwords.extend(add)
        return self
    
    def remove(self, *stopwords):
        """Remove one or more stopwords."""
        assert all(type(s) is str for s in stopwords), "All stopwords must be strings."
        exclude = [s for s in stopwords if s in self.stopwords]
        print(f"{len(exclude)} stopword(s) removed.")
        self.stopwords =  [s for s in self.stopwords if not s in exclude]
        return self
                        
    def __is_stopword__(self, sw: str) -> bool:
        """User interaction for identifying a stopword."""
        q = input(f"Keep \033[1m'{sw}'\033[0m (y/n)? ").lower().strip()
        if not q in 'yn' or len(q) != 1:
            print("\t\t\tInvalid input.")
            return self.__is_stopword__(sw)
        else:
            return q == 'y'
    
    def default_different(self) -> list:
        """Get the different stopwords from the default ones."""
        return list(set(self.stopwords) - set(self.default))
    
    def default_missing(self) -> list:
        """Get the default stopwords not present in the current ones."""
        return list(set(self.default) - set(self.stopwords))
    
    def save(self, verbose=1):
        """Save current stopwords in current working directory as a pickle file adding a datetime identifier into the filename."""
        fp = 'stopwords{}.pkl'.format(util.now())
        return util.pickle_save(self.stopwords, fp, verbose)
    
    def load(self, fp, verbose=1):
        """
        Replace current stopwords with a pickle file.
        
        Parameters
        ----------
        fp: str
           The filename of a pickle file.
        
        verbose: int or bool
            Whether to print the loaded stopwords. Default: 1.
        """
        obj = util.pickle_load(fp, verbose)
        assert util.is_iter(obj), "Loaded object must be an iterable."
        assert all(type(s) is str for s in obj), "Loaded sequence must contain only strings."
        self.stopwords = list(obj)
        return self
    
    def __repr__(self):
        return f"<Stopwords(length={self.len})>"
        
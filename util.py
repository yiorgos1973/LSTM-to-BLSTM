import os
import pickle
import datetime

from collections import OrderedDict


def now():
    """
    Get a string with current time now containing only numbers. 
    """
    now = datetime.datetime.now()
    return "{:4}{:2}{:2}{:2}{:2}{:2}".format(now.year, now.month, now.day, now.hour, now.minute, now.second).replace(' ', '0')


def pickle_save(obj, fp, verbose=1):
    """Save an object as a pickle file."""
    try:
        with open(fp, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as err:
        if verbose: print("Saving object failed. Error:", err)
        return False
    else:
        if verbose: print(f"Object saved as '{fp}'")
        return True

    
def pickle_load(fp, verbose=1):
    """Load a pickle file."""
    try:
        with open(fp, 'rb') as f:
            obj = pickle.load(f)
    except Exception as err:
        if verbose: print(f"Loading '{fp}' failed. Error:", err)
        return None
    else:
        if verbose: print(f"Object loaded successfully.")
        return obj
    

def is_iter(obj):
    """Check whether an object is iterable."""
    try:
        iter(obj)
        return True
    except:
        return False
    

def sort_dict(di, by='values', reverse=True):
    """
    Sort a dictionary by values or keys.
    """
    assert by in ['values', 'keys'], "Parameter 'by' can be 'values' or 'keys'"
    keys = sorted(di, key=lambda t: di[t] if by == 'values' else t, reverse=reverse)
    return OrderedDict((k, di[k]) for k in keys)
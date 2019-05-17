"""
Created on Wed Oct 11 16:57:59 2017

@author: tomas
"""

import numpy as np

np.errstate(divide='ignore', invalid='ignore')


def getopt(opt, key, default=None):
    if key in opt:
        return opt[key]
    elif default != None:
        return default
    else:
        raise ValueError("No default value provided for key %s" % key)


def ensureopt(opt, key):
    assert key in opt


def build_vocab(data):
    """ Builds a set that contains the vocab."""
    texts = []
    for datum in data:
        for r in datum['regions']:
            texts.append(r['label'])

    # vocab, indeces = np.unique(texts, return_index=True)
    vocab = np.unique(texts)
    return vocab  # , indeces

"""
Common functions for loading datasets etc.

28 May 2016
goker erdogan
https://github.com/gokererdogan
"""

import gzip
import cPickle as pkl

def load_mnist(path='./datasets'):
    """
    Load MNIST data from disk.
    Data can be downloaded from http://deeplearning.net/data/mnist/mnist.pkl.gz

    Returns
        (2-tuple): training set consisting of training data and class labels
        (2-tuple): validation set consisting of validation data and class labels
        (2-tuple): test set consisting of test data and class labels
    """
    f = gzip.open('{0:s}/mnist.pkl.gz'.format(path), mode='rb')
    train_set, val_set, test_set = pkl.load(f)
    f.close()
    return train_set, val_set, test_set

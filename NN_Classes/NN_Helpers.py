import numpy as np

def one_hot(Y):
    Y = Y.astype(int) # one_hot_Y = np.zeros(( Y.size, (np.max(Y) - np.min(Y) + 1) ))=      # = array with dimensions: [ (n) , (# of classes, which is 10) ]
    one_hot_Y = np.zeros(( Y.size, len(np.unique(Y)) ))   # = array with dimensions: [ (n) , (# of classes, which is 10) ]
    one_hot_Y[np.arange(Y.size), Y - np.min(Y)] = 1
    return one_hot_Y
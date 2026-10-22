import numpy as np

def compute_accuracy(logits, targets):
    predictions = np.argmax(logits, axis=0)
    # print ("predictions.shape", predictions.shape, "\ntargets.shape", targets.shape)
    # true_labels = np.argmax(targets, axis=1)
    assert predictions.shape == targets.shape
    return np.mean(predictions == targets)
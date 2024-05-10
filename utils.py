import numpy as np


def get_pruning_rate(weights):
    int_weights = []
    for w in weights:
        w = np.array(w)
        int_weights.extend(w.astype(bool).astype(int).flatten())
    pruning_rate = np.mean(int_weights)
    pruning_rate = 1 - pruning_rate
    return pruning_rate

def get_pruning_rates(weights):
    pruning_rates = []
    for w in weights:
        pruning_rates.append(get_pruning_rate([w]))
    return pruning_rates

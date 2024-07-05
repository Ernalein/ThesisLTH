import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# pruning rates

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

# number of parameters

def get_params_count(weights):
    n_weights = []
    for w in weights:
        w = np.array(w)
        n_weights.append(len(list(w.flatten())))
    count = np.sum(n_weights)
    return count

def get_params_counts(weights):
    counts = []
    for w in weights:
        counts.append(get_params_count([w]))
    return counts


# number of unpruned parameters

def get_unpruned_params_count(weights):
    int_weights = []
    for w in weights:
        w = np.array(w)
        int_weights.extend(w.astype(bool).astype(int).flatten())
    count_unpruned = np.sum(int_weights)
    return count_unpruned

def get_unpruned_params_counts(weights):
    counts_unpruned = []
    for w in weights:
        counts_unpruned.append(get_unpruned_params_count([w]))
    return counts_unpruned

# arrays
    
def get_flat(arrays):
    flat = []
    for array in arrays:
        flat.extend(array.flat)
    return flat

# plotting

def plot_losses(datasetname, pruning_name, losses, title):
    fig= plt.figure(figsize=(10,6))
    plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel("Loss/Accuracy")
    for key in losses.keys():
         plt.plot(losses[key],label=key)
    plt.legend()
    plt.savefig(f"3b Plots/{datasetname}_{pruning_name}_losses.png")
    plt.show()


def pretty_coll_sign_distr_plot(coll_sign_distr, alpha = 0.075):

    g = sns.PairGrid(coll_sign_distr, height=1.75)
    g.map_lower(sns.scatterplot,alpha = alpha, palette="rocket", color="#421B45")
    g.map_upper(sns.kdeplot, fill=True, cmap="rocket")
    g.map_diag(sns.histplot, color="#A3195B", element="step")

    return g
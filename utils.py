import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    
def get_flat(arrays):
    flat = []
    for array in arrays:
        flat.extend(array.flat)
    return flat

def pretty_coll_sign_distr_plot(coll_sign_distr, alpha = 0.075):

    g = sns.PairGrid(coll_sign_distr, height=1.75)
    g.map_lower(sns.scatterplot,alpha = alpha, palette="rocket", color="#421B45")
    g.map_upper(sns.kdeplot, fill=True, cmap="rocket")
    g.map_diag(sns.histplot, color="#A3195B", element="step")

    return g
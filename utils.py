import numpy as np
import matplotlib.pyplot as plt


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
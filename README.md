# LTHSignDistributions
The repository contains the code for my Bachelor Thesis about Sign Distributions in Winning Tickets.

## Contents

The repository is structured chronologically according to the steps of the experiments.

Main Experiment:  
- `1a Creating WTs with IMP.ipynb` creates Winning Tickets using Iterative Magnitude Pruning
- `1b WTs/` is the folder where the weights of the Winning Tickets are stored.
- `2a Extracting sign distributions.ipynb` loads the generated Winning Tickets to extract the Sign distributions
- `2b Sign distributions/` contains the files with the raw data of the extracted sign distributions
- `3a Plotting sign distributions.ipynb` loads the extracted sign Distributions and plots them in different ways
- `3b Plots/` contains the sign distribution plots
- `4a Similarities.ipynb` calculates the similarities between conditions and performs t-tests
- `4b Similarities/` holds the measures similarities and the results of the t-tests
- `4c Comparing Similarities.ipynb` plots the similarities and prints the tables with the t-test results

Further Analysis:
- `5a Clustering.ipynb` clusters the cumulative Sign Distributions
- `5b Clusters/` holds the cluster labels for the cumulative Sign Distributions
- `5c Attribution methods.ipynb` calculates contribution scores of and plots them
- `5d Contribution scores/` holds the contribution scores measured and the corresponding plots
- `5e Clustering and Contributions.ipynb` combines the clusters and contribution scores in a table
- `6a Training plots.ipynb` trains all Winning Tickets, Random Tickets and Control Models on each dataset and plots the accuracies
- `6b Training statistics/` contains the measured training statistics and the resulting plot

Other:
- `cnn_architecture.py` holds the Conv-2 architecture used in this study
- `load_datasets.py` holds functions for loading each of the three datasets used in this study
- `utils.py` holds useful functions for plotting and accessing certain parameters

## Requirements

To run the code, create an environment using the `environment.yml` file. Also load the SVHN (http://ufldl.stanford.edu/housenumbers/) and CINIC-10 (https://datashare.ed.ac.uk/handle/10283/3192) datasets to build the following folder structure:

- `data/`
  - `CIFAR/`...
  - `CINIC-10`
    - `test/`...
    - `train/`...
  - `SVHN/`
    - `test_32x32.mat`
    - `train_32x32.mat`
   
The CIFAR-10 dataset will be loaded into the corresponding file when runing the code via tensorflow-datasets.

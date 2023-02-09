import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.spn_utils import denormalize

def create_hist(data, bins,  title, xlabel, ylabel):
    """
    Creates a plt histogram with a given data, bins, title, xlabel, ylabel.
    """

    plt.clf()
    plt.hist(data, bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def spn_label_distrubution(samples_df, savepath, binwidth):
    """
    Creates a histogram of the data labels (dperf, sparsity) with given bins and saves the figures.
    :param samples_df: summary dataframe
    :param savepath: path to the folder where the resulted figures should be saved
    :param binwidth: bin width
    :return:
    """

    bins = np.arange(0, 1, binwidth)
    create_hist(denormalize(samples_df['spars'].tolist(), 0 ,1), bins, "Distribution of Label: sparsity", "sparsity", "number of samples" )
    plt.savefig(os.path.join(savepath, "label_spars_distribution.png"))

    create_hist(denormalize(samples_df['dmap'].tolist(), 0 ,1), bins, "Distribution of Label: dmap", "dmap", "number of samples")
    plt.savefig(os.path.join(savepath, "label_dmap_distribution.png"))


def create_summary_dataframe(labelspath): #, savepath, binwidth, threshold):
    """
    Creates a summary datframe of the samples.
    :param labelspath: path to the labels folder
    :return: dataframe with the following information:
             "spars", "dmap", "drec", "dprec", "file", "last pruned layer"
    """

    state_names = ['alpha', 'in_ch', 'out_ch', 'kernel', 'stride', 'pad', 'spars']
    samples_df = pd.DataFrame(columns=["spars", "dmap", "drec", "dprec", "file", "last pruned layer"])

    samples = os.listdir(labelspath)

    for sample in samples:
        with open(os.path.join(labelspath, sample), "r") as f:
            spars, dmap, drec, dprec = f.read().split(" ")

        """ This way the dataframe doesn't contain the data to the first layer, but we don't really care about 
               that in this task."""
        state_df = pd.read_csv(os.path.join(labelspath.replace("labels","states"), sample), sep=" ", names=state_names)

        # Get the fist row where kernel is not -1
        last_pruned_layer = state_df[(state_df.kernel > -1.0)].index[-1] + 1

        samples_df.loc[len(samples_df)] = [spars, dmap, drec, dprec, sample, last_pruned_layer]

    convert_dict = {'spars': float, 'dmap': float, 'drec': float, 'dprec': float, 'file': str, 'last pruned layer': int}
    samples_df = samples_df.astype(convert_dict)
    print(samples_df)

    return samples_df



if __name__ == '__main__':

    datapath = "/data/blanka/DATASETS/SPN/all4/labels"
    path = "/home/blanka/ERLPruning/sandbox"

    samples_df = create_summary_dataframe(datapath)
    spn_label_distrubution(samples_df, path, binwidth=0.1)

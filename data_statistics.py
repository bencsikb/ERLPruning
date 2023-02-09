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


def spn_label_distrubution(datapath, savepath, binwidth):
    """
    Create a histogram of the data labels (dperf, sparsity) with given bins.
    :param datapath: label folder
    :param savepath: path to the folder where the resulted figures should be saved
    :param binwidth: bin width
    :return:
    """

    samples = os.listdir(datapath)
    samples_df = pd.DataFrame(columns=["spars", "dmap", "drec", "dprec", "file"])

    for sample in samples:
        with open(os.path.join(datapath, sample), "r") as f:
            spars, dmap, drec, dprec = f.read().split(" ")
            samples_df.loc[len(samples_df)] = [spars, dmap, drec, dprec, sample]

        #df = pd.read_csv(os.path.join(datapath, sample), sep=" ")
        #df[sample] = 0
        #samples_df.loc[len(samples_df)] = df.columns

        convert_dict = {'spars': float, 'dmap': float, 'drec':float,  'dprec': float, 'file': str}
        samples_df = samples_df.astype(convert_dict)


    bins = np.arange(0, 1, binwidth)
    create_hist(denormalize(samples_df['spars'].tolist(), 0 ,1), bins, "Distribution of Label: sparsity", "sparsity", "number of samples" )
    plt.savefig(os.path.join(savepath, "label_spars_distribution.png"))

    create_hist(denormalize(samples_df['dmap'].tolist(), 0 ,1), bins, "Distribution of Label: dmap", "dmap", "number of samples")
    plt.savefig(os.path.join(savepath, "label_dmap_distribution.png"))


def spn_label_distrubution_wcondition(datapath, savepath, binwidth, threshold):
    """

    :param datapath:
    :param savepath:
    :param binwidth:
    :param threshold:
    :return:
    """

    state_samples = os.listdir(datapath)

    for state_sample in state_samples:
        print()


        



if __name__ == '__main__':

    datapath = "/data/blanka/DATASETS/SPN/all4/labels"
    path = "/home/blanka/ERLPruning/sandbox"
    spn_label_distrubution(datapath, path, binwidth=0.1)
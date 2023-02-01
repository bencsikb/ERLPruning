import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

import argparse
import math
import os
import random
import time


def denormalize(x, x_min, x_max):
    # Denormalize ground truth and predicted error

    if torch.is_tensor(x):
        ret = ((x + 1) * (x_max - x_min)) / 2 + x_min
    elif len(x):
        print("list")
        ret = [((xi + 1) * (x_max - x_min)) / 2 + x_min for xi in x]
    else:
        print("single")
        x = float(x)
        ret = ((x + 1) * (x_max - x_min)) / 2 + x_min

    return ret


def calc_metrics(gt, pred, margin=0.05):
    """
    Calculate standard regression metrics from sklearn and customized metrics for evaluating SPN.
    :param gt: ground truth - tensor
    :param pred: predicted values - tensor
    :param margin: for calculating accuracy with a margin - denormalized value
    :return: list of the calculated metrics
            - margin_accuracy: proportion of the denormalized predictions falling into the margin
            - negsign_recall: proportion of the negative gts predicted with a negative sign as well
                              (in dperf the negative sign means performance improvement)
            - mean_absoute_error: with sklearn on denormalized data
            - max_error: with sklearn on denormalized data
            - mean_squared_error: with sklearn on denormalized data
            - r2_score: with sklearn on denormalized data
    """

    metrics = []
    gt, pred = denormalize(gt, 0.0, 1.0), denormalize(pred, 0.0, 1.0)

    # handcrafted metrics

    margin_bool = torch.abs(gt - pred) <= margin  # tensor with bool values
    margin_accuracy = margin_bool.float().mean()
    metrics.append(margin_accuracy.item())

    negsign_bool = (gt < 0.0)
    nmb_negsign_gt = torch.count_nonzero(negsign_bool)
    indices = torch.nonzero(negsign_bool)
    pred_negsign_bool = pred[indices] < 0.0
    nmb_negsign_pred = torch.count_nonzero(pred_negsign_bool)
    negsign_recall = nmb_negsign_pred / (nmb_negsign_gt + 1e-15)
    metrics.append(negsign_recall.item())


    # sklearn regression metrics
    gt, pred = gt.cpu().detach().numpy(), pred.cpu().detach().numpy()

    metrics.append(mean_absolute_error(gt, pred))
    metrics.append(max_error(gt, pred))
    metrics.append(mean_squared_error(gt, pred))
    metrics.append(r2_score(gt, pred))
    """
    metrics.append(negsign_recall.item())
    metrics.append(negsign_recall.item())
    metrics.append(negsign_recall.item())
    metrics.append(negsign_recall.item())
    """




    return metrics
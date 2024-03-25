import torch
from torch import nn
import numpy as np
import sklearn


def denormalize(x, x_min, x_max):
    # Denormalize ground truth and predicted error

    if torch.is_tensor(x):
        ret = ((x + 1) * (x_max - x_min)) / 2 + x_min
        ret = torch.clamp(ret, 0, 1)
    elif isinstance(x, list):
        ret = [((float(xi) + 1) * (x_max - x_min)) / 2 + x_min for xi in x]
        ret = [max(min(xi, 1), 0) for xi in ret]
    else:
        x = float(x)
        ret = ((x + 1) * (x_max - x_min)) / 2 + x_min
        ret = max(min(ret, 1), 0)

    return ret



def normalize(x, x_min, x_max):
    # Between -1 and 1

    if torch.is_tensor(x):
        ret = 2 * ((x - x_min) / (x_max - x_min)) - 1
    elif type(x) == list:
        ret = [2 * ((float(xi) - x_min) / (x_max - x_min)) - 1 for xi in x]
    else:
        x = float(x)
        ret = 2 * ((x - x_min) / (x_max - x_min)) - 1

    return ret
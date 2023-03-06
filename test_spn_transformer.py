import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import argparse
import math
import os
import random
import time

from utils.spn_utils import denormalize, calc_metrics


def validate(dataloader, model, criterion_dperf, criterion_spars, margin, device="cuda"):

    model.eval()
    metrics_sum_dperf = np.zeros(6)
    metrics_sum_spars = np.zeros(6)

    running_loss = 0
    cnnt = 0

    for batch_i, (data, label_gt) in enumerate(dataloader):
        data = data.type(torch.float32).cuda()
        data = torch.cat((data[:, :44], data[:, 264:]), dim=1)
        label_gt = label_gt.type(torch.float32).cuda()
        data = data.unsqueeze(dim=2)  # embedding size --> 1
        label_gt = label_gt.unsqueeze(dim=2)
        #print(f"datashape {data.shape}, labelshape {label_gt.shape}")

        sequence_length = label_gt.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        prediction = model(data, label_gt, tgt_mask)
        prediction = prediction.permute(1, 0, 2)

        loss = criterion_dperf(denormalize(label_gt[:, 0], 0, 1), denormalize(prediction[:, 0], 0, 1)) + criterion_spars(denormalize(label_gt[:, 1], 0, 1), denormalize(prediction[:, 1], 0, 1))
        running_loss += loss.cpu().item()
        metrics_sum_dperf += calc_metrics(label_gt[:, 0], prediction[:, 0], margin=margin)
        metrics_sum_spars += calc_metrics(label_gt[:, 1], prediction[:, 1], margin=margin)

    # Calculate validation metrics

    running_loss /= len(dataloader)
    metrics_avg_dperf = metrics_sum_dperf / len(dataloader)
    metrics_avg_spars = metrics_sum_spars / len(dataloader)


    return running_loss, metrics_avg_dperf, metrics_avg_spars
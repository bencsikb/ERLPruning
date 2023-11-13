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
import yaml

from utils.spn_utils import denormalize, calc_metrics
from utils.datasets import create_pruning_dataloader
from utils.general import optimizer_to, scheduler_to
from utils.config_parser import ConfigParser


def validate(dataloader, model, criterion_dperf, criterion_spars, margin, device):

    model.eval()
    metrics_sum_dperf = np.zeros(6)
    metrics_sum_spars = np.zeros(6)

    running_loss = 0
    cnnt = 0

    for batch_i, (data, label_gt) in enumerate(dataloader):
        data = data.type(torch.float32).to(device)
        data = torch.cat((data[:, :, 0], data[:, :, -1]), dim=1).to(device)  # use only alpha and spars as state features
        label_gt = label_gt.type(torch.float32).to(device)
        #print(f"datashape {data.shape}, labelshape {label_gt.shape}")

        prediction = model(data)
        loss = criterion_dperf(denormalize(label_gt[:, 1], 0, 1), denormalize(prediction[:, 0], 0, 1)) \
               + criterion_spars(denormalize(label_gt[:, 0], 0, 1), denormalize(prediction[:, 1], 0, 1))
        running_loss += loss.cpu().item()
        metrics_sum_dperf += calc_metrics(label_gt[:, 1], prediction[:, 0], margin=margin)
        metrics_sum_spars += calc_metrics(label_gt[:, 0], prediction[:, 1], margin=margin)

    # Calculate validation metrics

    running_loss /= len(dataloader)
    metrics_avg_dperf = metrics_sum_dperf / len(dataloader)
    metrics_avg_spars = metrics_sum_spars / len(dataloader)


    return running_loss, metrics_avg_dperf, metrics_avg_spars


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='spn')
    parser.add_argument('--device', default='')
    opt = parser.parse_args()

    conf = ConfigParser.prepare_conf(opt)
    if len(opt.device):
        device = opt.device
    else:
        device = conf.train.device
    
    dataloader_val, dataset_val = create_pruning_dataloader(conf.data.data_path, conf.data.val_ids,  conf.data.cache_path, conf.data.cache_ext+"_val", batch_size=conf.train.batch_size)


    if conf.model.pretrained:
        ckpt = torch.load(os.path.join(conf.paths.model_dir, conf.model.pretrained))
        epoch = ckpt['epoch']
        model = ckpt['model'].to(device)
        if conf.model.old:
            criterion_dperf = ckpt['criterion_err'].to(device) 
        else:
            criterion_dperf = ckpt['criterion_dperf'].to(device)      
        criterion_spars = ckpt['criterion_spars'].to(device)
        lr_sched = ckpt['scheduler']
        optimizer = ckpt['optimizer']
        optimizer_to(optimizer, device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        for g in optimizer.param_groups:
           g['lr'] = 0.00008
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.train.epochs, eta_min=0.00001, last_epoch=epoch)
        scheduler_to(lr_sched, device)
        lr_sched.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % lr_sched.get_lr()[0]
        print("pretrained", epoch)
        print(lr_print)
    else:
        print("Error! Pretrained model is missing..")
    
    val_running_loss, val_metrics_avg_dperf, val_metrics_avg_spars = validate(dataloader_val, model,
                                                                                    criterion_dperf, criterion_spars, conf.train.margin, device)

    print(f"Validation results for {conf.model.pretrained}:")
    print(val_running_loss, val_metrics_avg_dperf, val_metrics_avg_spars)
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
    parser.add_argument('--data', type=str, default='data/spndata.yaml', help='data.yaml path')
    parser.add_argument('--logdir', type=str, default='/nas/blanka_phd/runs/SPN', help='tensorboard log path')
    parser.add_argument('--cfg', type=str, default='cfg/spn.cfg')
    parser.add_argument('--device', default='cuda:1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--pretrained', type=str, default='/nas/blanka_phd/Models/SPN/fromscratch_coco_2/weights/last.pt')
    #parser.add_argument('--pretrained', type=str, default='/nas/blanka_phd/Models/SPN/test_97_2534.pth')
    parser.add_argument('--smalldata', type=bool, default=False)
    parser.add_argument('--test-case', type=str, default='trial0')

    parser.add_argument('--epochs', type=int, default=6000)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--margin', type=int, default=0.02)

    opt = parser.parse_args()



    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    # Validation data
    data_path_val = data_dict['all_state']
    label_path_val = data_dict['all_label']
    dataloader_val, dataset_val = create_pruning_dataloader(data_path_val, label_path_val, batch_size=opt.batch_size)

    if opt.pretrained:
        ckpt = torch.load(opt.pretrained)
        epoch = ckpt['epoch']
        model = ckpt['model'].to(opt.device)
        criterion_dperf = ckpt['criterion_dperf'].to(opt.device) # !!! if loading an old model, this is called criterion_err, otherwise criterion_dperf!!!
        criterion_spars = ckpt['criterion_spars'].to(opt.device)
        lr_sched = ckpt['scheduler']
        optimizer = ckpt['optimizer']
        optimizer_to(optimizer, opt.device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        for g in optimizer.param_groups:
           g['lr'] = 0.00008
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.00001, last_epoch=epoch)
        scheduler_to(lr_sched, opt.device)
        lr_sched.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % lr_sched.get_lr()[0]
        print("pretrained", epoch)
        print(lr_print)
    else:
        print("Error! Pretrained model is missing..")
    
    val_running_loss, val_metrics_avg_dperf, val_metrics_avg_spars = validate(dataloader_val, model,
                                                                                    criterion_dperf, criterion_spars, opt.margin, opt.device)

    print(f"Validation results for {opt.pretrained}:")
    print(val_running_loss, val_metrics_avg_dperf, val_metrics_avg_spars)
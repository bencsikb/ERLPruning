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
from pathlib import Path
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import create_pruning_dataloader
from utils.spn_utils import denormalize, calc_metrics
from utils.config_parser import ConfigParser
from models.error_pred_network import errorNet, errorNet2
import torch.utils.data
from utils.torch_utils import init_seeds as init_seeds_manual
from utils.general import optimizer_to, scheduler_to
from models.models import *
from utils.logger import BasicLogger
#from utils.losses import LogCoshLoss, NegativeWeightedMSELoss
from utils.losses import LogCoshLoss
from utils.optimizers import RAdam, Lamb
from test_spn import validate

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def train(model, optimizer, lr_sched, conf, epoch, device, dataloader, dataloader_val, tb_writer=None):

    # Path for saving weights
    log_dir = Path(tb_writer.log_dir)
    wdir = str(log_dir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = str(log_dir / 'results.txt')
    epochs, batch_size =  conf.train.epochs, conf.train.batch_size,
    init_seeds_manual(42)

    # Save settings and model
    settings_dict = {"criterion_dperf": str(criterion_dperf), "criterion_spars": str(criterion_spars),  "optimizer": str(optimizer)}
    txt_logger.log_settings(conf, settings_dict)
    txt_logger.log_model(model)

    losses, errors, precisions, sign_precisions = [], [], [], []
    losses_val, errors_val, precisions_val, sign_precisions_val = [], [], [], []
    bestLoss = 10000

    while epoch < epochs:  
        """          
        model.train()
        print(f"epoch {epoch}")

        metrics_sum_dperf = np.zeros(6)
        metrics_sum_spars = np.zeros(6)

        running_loss = 0
        cnnt = 0

        print("len dataloader", len(dataloader))

        for batch_i, (data, label_gt) in enumerate(dataloader):
            print(f"batch {batch_i}")
            data = data.type(torch.float32).to(device)
            label_gt = label_gt.type(torch.float32).to(device)
            optimizer.zero_grad()
            data = torch.cat((data[:, :, 0], data[:, :, -1]), dim=1).to(device)  # use only alpha and spars as state features
            print(f"datashape {data.shape}, labelshape {label_gt.shape}")
            print(data.device)
            prediction = model(data)

            loss = criterion_dperf(denormalize(label_gt[:, 1], 0, 1),  denormalize(prediction[:, 0], 0, 1)) \
                   + criterion_spars(denormalize(label_gt[:, 0], 0, 1), denormalize(prediction[:, 1], 0, 1))
            # loss = criterion_err(label_gt[:,0], prediction[:,0]) + criterion_spars(label_gt[:,1], prediction[:,1])

            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item()
            # err, prec, neg_err, neg_corr, negsign_prec = calc_precision(error_gt, error_pred)
            metrics_sum_dperf += calc_metrics(label_gt[:, 1], prediction[:, 0], margin=conf.train.margin)
            metrics_sum_spars += calc_metrics(label_gt[:, 0], prediction[:, 1], margin=conf.train.margin)
            #print(metrics_sum_dperf[0,0], tmp[2]) #error
            #print(metrics_sum_dperf , tmp[0]) #accuracy
            #print(metrics_sum_dperf[0,3], tmp[1], "\n") #negsign recall
            #print(metrics_sum_dperf)
            #print(tmp)

            # print(batch_i, error_gt)

            # if error_gt[0].item() < -0.8 and data[0][46]==-1:
            # if error_gt[0].item() < -0.8 or error_gt[1].item() < -0.8 :

            # print(data.view(-1,44))
            print("Ground truth: ", label_gt[0,:])
            print("Predicted: ", prediction[0,:])
            # cnnt += 1
            # print("Error, prec", err, prec)

            # print(f"{batch_i}/{len(dataloader)} batches done")

        # print("cnt ", cnnt)

        # Calculate training metrics
        running_loss /= len(dataloader)
        metrics_avg_dperf = metrics_sum_dperf / len(dataloader)
        metrics_avg_spars = metrics_sum_spars / len(dataloader)


        # metrics_avg[0, 2:4] = metrics_sum[0, 2:4] /len(dataloader)   # error, precision
        # gt_negatives = metrics_sum[0, 7]
        # print(gt_negatives)
        # metrics_avg[0, 4:7] = metrics_sum[0, 4:7]/gt_negatives   # neg_error, negsign_hits, negsign_precision
        """

        # VALIDATION

        if epoch % conf.train.val_interval == 0:
            val_running_loss, val_metrics_avg_dperf, val_metrics_avg_spars = validate(dataloader_val, model,
                                                                                    criterion_dperf, criterion_spars, conf.train.margin, device)

        checkpoint = {'epoch': epoch,
                      'model': model,
                      'criterion_dperf': criterion_dperf,
                      'criterion_spars': criterion_spars,
                      'optimizer': optimizer,
                      'scheduler': lr_sched}

        print(val_running_loss, val_metrics_avg_dperf, val_metrics_avg_spars )

        
        # torch.save(checkpoint, os.path.join(opt.ckpt_save_path, opt.test_case + '.pth'))
        torch.save(checkpoint, last)

        # Save model
        if running_loss < bestLoss:
            bestLoss = running_loss
            torch.save(checkpoint, best)

        # Tensorboard logging
        tb_writer.add_scalar("train_loss", running_loss, epoch)
        tb_writer.add_scalar("val_loss", val_running_loss, epoch)

        tags_dperf_train = [f"dperf/train/margin_{conf.train.margin}_accuracy", "dperf/train/negsign_recall",  "dperf/train/mean_abs_error", "dperf/train/max_error", "dperf/train/mean_squared_error", "dperf/train/r2_score"]
        tags_dperf_val  = [x.replace("train", "val") for x in tags_dperf_train]
        tags_spars_train  = [x.replace("dperf", "spars") for x in tags_dperf_train]
        tags_spars_val  = [x.replace("train", "val") for x in tags_spars_train]

        for tag, scalar in zip(tags_dperf_train, metrics_avg_dperf):
            tb_writer.add_scalar(tag, scalar, epoch)

        for tag, scalar in zip(tags_dperf_val, val_metrics_avg_dperf):
            tb_writer.add_scalar(tag, scalar, epoch)

        for tag, scalar in zip(tags_spars_train, metrics_avg_spars):
            tb_writer.add_scalar(tag, scalar, epoch)

        for tag, scalar in zip(tags_spars_val, val_metrics_avg_spars):
            tb_writer.add_scalar(tag, scalar, epoch)


        # Save params

        with open(results_file, 'a') as f:
            f.write(F"{epoch} {running_loss} {val_running_loss} | {metrics_avg_dperf} | {val_metrics_avg_dperf} "
                    F"| {metrics_avg_spars} | {val_metrics_avg_spars}\n".replace("[", "").replace("]", "").replace(",","").replace("\t", ""))

        lr_sched.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % lr_sched.get_lr()[0]
        print(lr_print)
        epoch += 1
        
    
    print(bestLoss)
    plt.plot(losses)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='spn')
    parser.add_argument('--device', default='')
    parser.add_argument('--test-case', type=str, default='trial0')
    opt = parser.parse_args()

    conf = ConfigParser.prepare_conf(opt)
    if len(opt.device):
        device = opt.device
    else:
        device = conf.train.device

    tb_writer = SummaryWriter(log_dir=os.path.join(conf.paths.log_dir, conf.logging.folder, opt.test_case ))
    txt_logger = BasicLogger(log_dir=os.path.join(conf.paths.log_dir, conf.logging.folder), test_case=opt.test_case)

    # Dataloaders
    dataloader, dataset = create_pruning_dataloader(conf.data.data_path, conf.data.train_ids, conf.data.cache_path, conf.data.cache_ext+"_train", batch_size=conf.train.batch_size)
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
        scheduler_to(lr_sched, conf.train.epochs)
        lr_sched.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % lr_sched.get_lr()[0]
        print("pretrained", epoch)
        print(lr_print)
    else:
        print("new model")
        epoch = 0
        model = errorNet2(88, 2).to(device)
        #model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.res_lr, weight_decay=1e-5)
        #optimizer = Lamb(model.parameters(), lr=0.001, weight_decay=1e-5)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.train.epochs, eta_min=0.000005, last_epoch=-1)
        criterion_dperf = LogCoshLoss().to(device)
        criterion_spars = LogCoshLoss().to(device) #nn.MSELoss().cuda()
        #criterion_err = NegativeWeightedMSELoss(5).cuda()
        #criterion_spars = torch.nn.MSELoss().cuda()

    print(model)

    train(model, optimizer, lr_sched, conf, epoch, device, dataloader, dataloader_val, tb_writer)





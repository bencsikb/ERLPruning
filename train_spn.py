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
from utils.errorPredNet_utils import denormalize, calc_precision
from models.error_pred_network import errorNet, errorNet2
import torch.utils.data
from models.models import *
#from utils.losses import LogCoshLoss, NegativeWeightedMSELoss
from utils.losses import LogCoshLoss
from utils.optimizers import RAdam, Lamb
from test_errorPredNet import validate

torch.manual_seed(42)
torch.cuda.manual_seed(42)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/pruningdata.yaml', help='data.yaml path')
    parser.add_argument('--logdir', type=str, default='runs/pruning_error', help='tensorboard log path')
    parser.add_argument('--ckpt-save-path', type=str, default='checkpoints/pruning_error_pred')
    parser.add_argument('--results-save-path', type=str, default='results/pruning_error_pred')
    parser.add_argument('--cfg', type=str, default='cfg/pruning_error_pred.cfg')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--smalldata', type=bool, default=True)
    parser.add_argument('--test-case', type=str, default='deleteme')
    #parser.add_argument('--test-case', type=str, default='test_90_rep_2')

    parser.add_argument('--epochNum', type=int, default=4000)
    parser.add_argument('--val_interval', type=int, default=5)
    #parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--batch-size', type=int, default=2048)

    opt = parser.parse_args()

    opt.pretrained = "checkpoints/pruning_error_pred/test_90_rep_2.pth"

    tb_writer = SummaryWriter(log_dir=os.path.join(opt.logdir, opt.test_case ))
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    # Trainloader
    data_path = data_dict['train_pruningdata']
    label_path = data_dict['train_label']
    dataloader, dataset = create_pruning_dataloader(data_path, label_path, batch_size=opt.batch_size)

    # Validation data
    data_path_val = data_dict['val_pruningdata']
    label_path_val = data_dict['val_label']
    dataloader_val, dataset_val = create_pruning_dataloader(data_path_val, label_path_val, batch_size=4)


    print(len(dataloader))


    if opt.pretrained:
        ckpt = torch.load(opt.pretrained)
        epoch = ckpt['epoch']
        model = ckpt['model']
        criterion_err = ckpt['criterion_err']
        criterion_spars = ckpt['criterion_spars']
        lr_sched = ckpt['scheduler']
        optimizer = ckpt['optimizer']
        for g in optimizer.param_groups:
           g['lr'] = 0.00008
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochNum, eta_min=0.00001,
                                                              last_epoch=epoch)
        lr_sched.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % lr_sched.get_lr()[0]
        print("pretrained", epoch)
        print(lr_print)
    else:
        print("new model")
        epoch = 0
        model = errorNet2(88, 2).cuda()
        #model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        #optimizer = Lamb(model.parameters(), lr=0.001, weight_decay=1e-5)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochNum, eta_min=0.000005,
                                                              last_epoch=-1)
        criterion_err = LogCoshLoss().cuda()
        criterion_spars = LogCoshLoss().cuda() #nn.MSELoss().cuda()
        #criterion_err = NegativeWeightedMSELoss(5).cuda()
        #criterion_spars = torch.nn.MSELoss().cuda()

    print(model)


    losses, errors, precisions, sign_precisions = [], [], [], []
    losses_val, errors_val, precisions_val, sign_precisions_val = [], [], [], []
    bestLoss = 10000


    while epoch < opt.epochNum:
        model.train()
        print(opt.test_case)

        metrics_sum_err = np.zeros((1,5))
        metrics_sum_spars = np.zeros((1,5))

        running_loss = 0
        cnnt = 0

        for batch_i, (data, label_gt) in enumerate(dataloader):

            data = data.type(torch.float32).cuda()
            label_gt = label_gt.type(torch.float32).cuda()
            optimizer.zero_grad()
            data = torch.cat((data[:,:44], data[:, 264:]), dim=1)
            prediction = model(data)
            loss = criterion_err(denormalize(label_gt[:,0],0,1), denormalize(prediction[:,0],0,1)) + criterion_spars(denormalize(label_gt[:,1],0,1), denormalize(prediction[:,1],0,1))
            #loss = criterion_err(label_gt[:,0], prediction[:,0]) + criterion_spars(label_gt[:,1], prediction[:,1])

            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item()
            #err, prec, neg_err, neg_corr, negsign_prec = calc_precision(error_gt, error_pred)
            metrics_sum_err += calc_precision(label_gt[:, 0], prediction[:,0], threshold=0.01)
            metrics_sum_spars += calc_precision(label_gt[:,1], prediction[:,1], threshold=0.01)


            #print(batch_i, error_gt)

            #if error_gt[0].item() < -0.8 and data[0][46]==-1:
            #if error_gt[0].item() < -0.8 or error_gt[1].item() < -0.8 :

                #print(data.view(-1,44))
                #print("Ground truth: ", error_gt)
                #print("Predicted: ", error_pred)
                #cnnt += 1
            #print("Error, prec", err, prec)

            #print(f"{batch_i}/{len(dataloader)} batches done")

        #print("cnt ", cnnt)


        # Calculate training metrics
        running_loss /= len(dataloader)
        metrics_avg_err = metrics_sum_err /  len(dataloader)
        metrics_avg_spars = metrics_sum_spars /  len(dataloader)

        """
        metrics_avg[0, 2:4] = metrics_sum[0, 2:4] /len(dataloader)   # error, precision
        gt_negatives = metrics_sum[0, 7]
        print(gt_negatives)
        metrics_avg[0, 4:7] = metrics_sum[0, 4:7]/gt_negatives   # neg_error, negsign_hits, negsign_precision
        """


        #VALIDATION

        if epoch % opt.val_interval == 0:
            val_running_loss, val_metrics_avg_err, val_metrics_avg_spars = validate(dataloader_val, model, criterion_err, criterion_spars)

        # Save model
        if running_loss < bestLoss:
            bestLoss = running_loss
            checkpoint = {'epoch': epoch,
                          'model': model,
                          'criterion_err': criterion_err,
                          'criterion_spars': criterion_spars,
                          'optimizer': optimizer,
                          'scheduler': lr_sched}

            torch.save(checkpoint, os.path.join(opt.ckpt_save_path, opt.test_case + '.pth'))


        # Tensorboard logging
        error = metrics_avg_err[0,0]
        precision = metrics_avg_err[0,1]
        tags = ['train/loss', 'train/error', 'train/precision']
        tb_writer.add_scalar(tags[0], running_loss, epoch)
        tb_writer.add_scalar(tags[1], error, epoch)
        tb_writer.add_scalar(tags[2], precision, epoch)

       # Save params

        # Save params

        # metrics_avg[:, 0] = epoch
        # metrics_avg[:, 1] = running_loss
        # print(F"error, precision, neg_error, negsign_hits, negsign_precision: {epoch}, {running_loss}, {metrics_avg_err[0, :]}, {metrics_avg_spars[0, :]}")

        if not os.path.exists(os.path.join(opt.results_save_path, opt.test_case)):
            print("created")
            os.mkdir(os.path.join(opt.results_save_path, opt.test_case))

        path = os.path.join(os.path.join(opt.results_save_path, opt.test_case), 'results.txt')
        with open(path, 'a') as f:
            # np.savetxt(f , metrics_avg, fmt='% 4f')
            f.write(F"{epoch}, {running_loss}, {metrics_avg_err[0, :]}, {metrics_avg_spars[0, :]} \n")

        if epoch % opt.val_interval == 0:
            # print(F"error, precision, neg_error, negsign_hits, negsign_precision: {epoch}, {val_running_loss}, {val_metrics_avg_err[0, :]}, {val_metrics_avg_spars[0, :]} \n")

            path = os.path.join(os.path.join(opt.results_save_path, opt.test_case), 'results_val.txt')
            with open(path, 'a') as f:
                # np.savetxt(f, val_metrics_avg, fmt='% 4f')
                f.write(F"{epoch}, {val_running_loss}, {val_metrics_avg_err[0, :]}, {val_metrics_avg_spars[0, :]} \n")

        lr_sched.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % lr_sched.get_lr()[0]
        print(lr_print)
        epoch += 1

    print(bestLoss)
    plt.plot(losses)
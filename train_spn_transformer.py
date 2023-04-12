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
from models.error_pred_network import errorNet, errorNet2
import torch.utils.data
from utils.torch_utils import init_seeds as init_seeds_manual
from models.models import *
from utils.logger import BasicLogger
#from utils.losses import LogCoshLoss, NegativeWeightedMSELoss
from utils.losses import LogCoshLoss
from utils.optimizers import RAdam, Lamb
from test_spn_transformer import validate
#from transformer import Transformer
from transformer_manual import Transformer #, get_tgt_mask

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def train(model, optimizer, lr_sched, opt, epoch, device, dataloader, dataloader_val, tb_writer=None):

    # Path for saving weights
    log_dir = Path(tb_writer.log_dir)
    wdir = str(log_dir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = str(log_dir / 'results.txt')
    opt_file = str(log_dir / 'logs.txt')
    model_file = str(log_dir / 'model.txt')
    epochs, batch_size =  opt.epochs, opt.batch_size,
    init_seeds_manual(42)

    # Save settings and model
    settings_dict = {"criterion_dperf": str(criterion_dperf), "criterion_spars": str(criterion_spars), "optimizer": str(optimizer)}
    txt_logger.log_settings(opt, settings_dict)
    txt_logger.log_model(model)


    losses, errors, precisions, sign_precisions = [], [], [], []
    losses_val, errors_val, precisions_val, sign_precisions_val = [], [], [], []
    bestLoss = 10000

    while epoch < epochs:
        model.train()
        print(f"epoch {epoch}")

        metrics_sum_dperf = np.zeros(6)
        metrics_sum_spars = np.zeros(6)
        max_error_dperf, max_error_spars = 0, 0

        running_loss = 0
        cnnt = 0

        print("len dataloader", len(dataloader))

        for batch_i, (data, label_gt) in enumerate(dataloader):
            # print(f"batch {batch_i}")

            print(f"data in train before cutting out features: {data.shape}")
            data = data.type(torch.float32).cuda() # [batch_size, n_layers*n_features]
            label_gt = label_gt.type(torch.float32).cuda()
            optimizer.zero_grad()

            #data = torch.cat((data[:, :107], data[:, -107:]), dim=1) #  TODO  use only alpha and spars as state features
            # data = data.unsqueeze(dim=2) # embedding size --> 1 [batch_size, n_features, embedding_size]
            #data = torch.cat((data[:, :, :1], data[:, :, -1:]), dim=2)
            data = torch.cat((data[:, :, :5], data[:, :, -1:]), dim=2)

            #label_gt = label_gt.unsqueeze(dim=2) # [batch_size, n_labels, 1]
            print(f"data in train after cutting out features: {data.shape}")

            tgt = torch.ones(label_gt.shape).cuda()
            print(f"data, ttgt {data.device} {tgt.device}")
            #prediction = model(data.unsqueeze(2), label_gt.unsqueeze(2))
            #final_linear = torch.nn.Linear(opt.dim_model, 2).to(opt.device)
            #prediction = final_linear(prediction[:, 0, :])
            prediction = model(data)
            prediction = prediction.permute(0, 1) # --> [batch_size, n_lables, sequence_length]
            print(f"label, prediction in train: {label_gt.shape}, {prediction.shape}")

            #loss = criterion_spars(denormalize(label_gt[:, 0, 0], 0, 1),  denormalize(prediction[:, 0, 0], 0, 1)) \
            #       + criterion_dperf(denormalize(label_gt[:, 1, 0], 0, 1), denormalize(prediction[:, 1, 0], 0, 1))

            loss = criterion_spars(denormalize(label_gt[:, 0], 0, 1), denormalize(prediction[:, 0], 0, 1)) \
                   + opt.dperf_loss_factor * criterion_dperf(denormalize(label_gt[:, 1], 0, 1), denormalize(prediction[:, 1], 0, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item()
            # err, prec, neg_err, neg_corr, negsign_prec = calc_precision(error_gt, error_pred)
            #metrics_sum_spars += calc_metrics(label_gt[:, 0, 0], prediction[:, 0, 0], margin=opt.margin)
            #metrics_sum_dperf += calc_metrics(label_gt[:, 1, 0], prediction[:, 1, 0], margin=opt.margin)
            metrics_spars = calc_metrics(label_gt[:, 0], prediction[:, 0], margin=opt.margin)
            metrics_dperf = calc_metrics(label_gt[:, 1], prediction[:, 1], margin=opt.margin)
            metrics_sum_spars +=  metrics_spars
            metrics_sum_dperf += metrics_dperf
            curr_max_error_spars = metrics_spars[3]
            curr_max_error_dperf = metrics_dperf[3]
            if curr_max_error_spars > max_error_spars : max_error_spars = curr_max_error_spars
            if curr_max_error_dperf > max_error_dperf : max_error_dperf = curr_max_error_dperf


        # Calculate training metrics
        running_loss /= len(dataloader)
        metrics_avg_dperf = metrics_sum_dperf / len(dataloader)
        metrics_avg_spars = metrics_sum_spars / len(dataloader)
        metrics_avg_dperf[3] = max_error_dperf
        metrics_avg_spars[3] = max_error_spars


        # VALIDATION

        if epoch % opt.val_interval == 0:
            val_running_loss, val_metrics_avg_dperf, val_metrics_avg_spars = validate(dataloader_val, model,
                                                                                    criterion_dperf, criterion_spars, opt.margin)

        checkpoint = {'epoch': epoch,
                      'model': model,
                      'criterion_dperf': criterion_dperf,
                      'criterion_spars': criterion_spars,
                      'optimizer': optimizer,
                      'scheduler': lr_sched}

        # torch.save(checkpoint, os.path.join(opt.ckpt_save_path, opt.test_case + '.pth'))
        torch.save(checkpoint, last)

        # Save model
        if running_loss < bestLoss:
            bestLoss = running_loss
            torch.save(checkpoint, best)

        # Tensorboard logging
        tb_writer.add_scalar("train_loss", running_loss, epoch)
        tb_writer.add_scalar("val_loss", val_running_loss, epoch)

        tags_dperf_train = [f"dperf/train/margin_{opt.margin}_accuracy", "dperf/train/negsign_recall",  "dperf/train/mean_abs_error", "dperf/train/max_error", "dperf/train/mean_squared_error", "dperf/train/r2_score"]
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
                    F"| {metrics_avg_spars} | {val_metrics_avg_spars}\n".replace("[", "").replace("]","").replace(",","").replace("\t", ""))

        lr_sched.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % lr_sched.get_lr()[0]
        print(lr_print)
        epoch += 1

    print(bestLoss)
    plt.plot(losses)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/spndata.yaml', help='data.yaml path')
    parser.add_argument('--logdir', type=str, default='/data/blanka/ERLPruning/runs/SPN', help='tensorboard log path')
    parser.add_argument('--results-save-path', type=str, default='results/pruning_error_pred')
    parser.add_argument('--cfg', type=str, default='cfg/spn.cfg')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--smalldata', type=bool, default=False)
    parser.add_argument('--test-case', type=str, default='manual_transformer_all54_05')
    #parser.add_argument('--test-case', type=str, default='test_90_rep_2')

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--val_interval', type=int, default=1)
    #parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--margin', type=float, default=0.02)

    # Transfromer params
    parser.add_argument('--nheads', type=int, default=2)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--dim_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dim_model', type=int, default=32)
    parser.add_argument('--dperf-loss-factor', default=1)
    parser.add_argument('--spars-loss-factor', default=1)


    opt = parser.parse_args()

    tb_writer = SummaryWriter(log_dir=os.path.join(opt.logdir, opt.test_case))
    txt_logger = BasicLogger(log_dir=opt.logdir, test_case=opt.test_case)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    # Trainloader
    data_path = data_dict['train_state']
    label_path = data_dict['train_label']
    dataloader, dataset = create_pruning_dataloader(data_path, label_path, batch_size=opt.batch_size)

    # Validation data
    data_path_val = data_dict['val_state']
    label_path_val = data_dict['val_label']
    dataloader_val, dataset_val = create_pruning_dataloader(data_path_val, label_path_val, batch_size=opt.batch_size)

    print("len dataloader", len(dataloader))

    if opt.pretrained:
        """
        ckpt = torch.load(opt.pretrained)
        epoch = ckpt['epoch']
        model = ckpt['model']
        criterion_dperf = ckpt['criterion_err']  # !!! if loading an old model, this is called criterion_err !!!
        criterion_spars = ckpt['criterion_spars']
        lr_sched = ckpt['scheduler']
        optimizer = ckpt['optimizer']
        for g in optimizer.param_groups:
            g['lr'] = 0.00008
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.00001,
                                                              last_epoch=epoch)
        lr_sched.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % lr_sched.get_lr()[0]
        print("pretrained", epoch)
        print(lr_print)
        """
    else:
        print("new model")
        epoch = 0
        """
        model = Transformer(
            # num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
            num_tokens =2, dim_model = opt.dim_model, num_heads = opt.nheads, num_encoder_layers = opt.nlayers, num_decoder_layers = opt.nlayers, dropout_p = 0.05
        ).to(opt.device)        # model.apply(init_weights)
        """
        model = Transformer(nhead=opt.nheads, nlayers=opt.nlayers, dim_model=opt.dim_model, dim_ff=opt.dim_ff,  out_size=2, dropout=opt.dropout).to(opt.device)
        #model = nn.Transformer(d_model=opt.dim_model, nhead=opt.nheads, num_encoder_layers=opt.nlayers, num_decoder_layers=opt.nlayers, dim_feedforward=opt.dim_ff, dropout=opt.dropout).to(opt.device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        optimizer = Lamb(model.parameters(), lr=0.001, weight_decay=1e-5)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.000005,
                                                              last_epoch=-1)
        criterion_dperf =  LogCoshLoss().cuda()
        criterion_spars =  LogCoshLoss().cuda()  # nn.MSELoss().cuda()
        # criterion_err = NegativeWeightedMSELoss(5).cuda()
        # criterion_spars = torch.nn.MSELoss().cuda()

    print(model)

    train(model, optimizer, lr_sched, opt, epoch, opt.device, dataloader, dataloader_val, tb_writer)


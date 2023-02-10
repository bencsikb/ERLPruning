import argparse
import torch.nn as nn
import torch
import numpy as np
import math
import os
import random
import time
from pathlib import Path
import torch_pruning as tp
import pandas as pd
import copy

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from prune_for_error import prune_network, normalize
from models.models import *
from utils.layers import *
from test import *
from utils.datasets import *
from utils.general import *

global_filewrite = False
global_print = False

glob_dims = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]

def normalize(x, x_min, x_max):
    # Between -1 and 1

    x = float(x)
    x = 2 * ((x - x_min) / (x_max - x_min)) - 1
    return x


def choose_channels_to_prune(layer, layer_idx, alpha, dim):
    """
    Selects the channels that need to be removed, given the condition.
    Always examines the output layers (dim = 0).
    :param layer:
    :param layer_idx:
    :param alpha:
    :param dim:
    :return: Indices of the channels to be removed.
    """

    # Calculate norm for every channel
    if dim == 0:
        norms = (torch.norm(layer.weight.data, 'fro', dim=[2, 3]))
        if global_print: print("Layer dimenzions: ", layer.weight.data.shape[2], layer.weight.data.shape[3])
        norms = torch.norm(norms, 'fro', dim=1)

    # Count the average and standard deviation of the channel norms in the layer
    norms_avg = torch.mean(norms)
    norms_std = torch.std(norms)

    if global_print: print("avg ", norms_avg)
    if global_print: print("std ", norms_std)

    # Find the indices of the channels that have to be pruned

    norm_condition = torch.logical_and((torch.absolute(norms) < norms_avg + alpha * norms_std),
                                      (torch.absolute(norms) > norms_avg - alpha * norms_std))

    indices = norm_condition.nonzero()

    if global_print and (layer_idx == 1):
        print(f"layer {layer}, alpha: {alpha}")
        print(f"layer shape: {layer.weight.shape} \nnorms shape: {norms.shape}")
        print(f"norms_avg: {norms_avg}, norms_std: {norms_std}")
        print(f"{norms_avg - alpha * norms_std} < norm < {norms_avg + alpha * norms_std}")
        print(norms)
        print(indexes.shape)

    # Avoid pruning the whole layer
    if (dim == 0) and (len(indices) == layer.out_channels):
        c = random.choice(np.arange(0, layer.out_channels - 1, 1).tolist())
        print("avoiding", c)
        # indexes.pop(c)
        indices = indices[1:]

    if alpha != 0: print("len(indexes) ", len(indices), layer, layer_idx, alpha, dim)

    return indices.squeeze(dim=1).tolist()

def prune_network(network, detection_layers, layer_to_prune, alpha_seq=None, state_size_check=False, device='cuda', dataset_make=False):
    """
    :param network: model to be pruned
    :param detection_layers: layers that can't be pruned (ex. YOLO Layers)
    :param layer_to_prune: only important when generating the dataset, at this layer the layer-related features are saved
    :param state_size_check: True if the function is used for determining the number of prunable layers in the beginning.
                            In this case, alpha_seq can be None.
    :param dataset_make: True if the automatic data generation is the case, False in case of training the RL agent.
    :return:
    """

    # if dim == 0 --> prune output channels
    # if dim == 1 --> prune input channels
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    layer_cnt = 0
    network_size = len(network.module_list)
    dims = [0] * 1
    output_filters = []
    index_lens = []

    parser = {'dim': dim,
              'in_ch': 0,
              'out_ch': 0,
              'kernel': 0,
              'stride': 0,
              'pad': 0,
              'nmb_of_pruned_ch': 0}

    DG = tp.DependencyGraph()
    DG.build_dependency(network, example_inputs=torch.randn(1, 3, 224, 224).to(device))
    pruning_groups = []

    save_path = f"/home/blanka/ERLPruning/sandbox/pruning_develop_data_{layer_to_prune}.txt"
    with open(save_path, 'w') as f:
    #if True:
        for i in range(network_size):

            start_time_forloop = time.time()

            sequential_size = len(network.module_list[i])
            module_def = network.module_defs[i]
            if global_filewrite: f.write("\nmodule_def " + str(module_def["type"]) + str(module_def) + "\n")

            if i in detection_layers:
                #print("Skipped, as it is a YOLO Layer.")
                continue

            if module_def["type"] == "convolutional":

                    layer = network.module_list[i][0] # if j == 0: nn.Conv2d instance (the rest is BacthNorm and activation)

                    if global_filewrite: f.write("Pruning layer " + str(i) + " " + str(layer) + "\n")

                    if isinstance(layer, nn.Conv2d): # and glob_dims[i] == 0: # only if testing old 44 long alpha_seqs

                        #alpha = float(torch.round(alpha_seq[layer_cnt]).item()) ## BUG !!!!
                        if state_size_check: alpha = 0
                        else:   alpha = np.around(alpha_seq[layer_cnt].item(), 1)

                        f.write(str(i) + str(layer) + "\n")

                        layer_cnt += 1
                        indices = choose_channels_to_prune(layer, layer_cnt, alpha, dim)

                        pruning_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)

                        if DG.check_pruning_group(pruning_group):
                            pruning_group.exec()

                        # Logging

                        if i == layer_to_prune:
                            parser['dim'] = dim
                            parser['in_ch'] = layer.in_channels
                            parser['out_ch'] = layer.out_channels
                            parser['kernel'] = layer.kernel_size[0]
                            parser['stride'] = layer.stride[0]
                            parser['pad'] = layer.padding[0]
                            parser['nmb_of_pruned_ch'] = len(indices)
                            # parser['channels_before'] = layer.in_channels if dim==1 else layer.out_channels

                        if global_filewrite: f.write("conv dims " + str(dims) + "\n")
                        # f.write("conv alphas"+str(alphas)+ "\n")
                        if global_filewrite: f.write("conv output_filters " + str(output_filters) + "\n")
                        if global_filewrite:  f.write("conv dim " + str(dim) + "\n")
                        if indices is not None:
                            if global_filewrite: f.write("conv indexes " + str(len(indexes)) + "\n")
                        else:
                            if global_filewrite: f.write("conv indexes " + str(0) + "\n")
                        if global_filewrite: f.write("conv index lens " + str(index_lens) + "\n")

                    else:
                        continue
                        #print(f"Error! The layer is not an instance of nn.Conv2d! \n{layer}")


    if dataset_make:
        return network, parser
    elif state_size_check:
        return layer_cnt
    else:
        return network

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # For data split
    parser.add_argument('--labelpath', type=str, default='/data/blanka/DATASETS/KITTI/original/label_2')
    parser.add_argument('--savepath', type=str, default='/home/blanka/ERLPruning/data')
    parser.add_argument('--train_idxpath', default="/home/blanka/ERLPruning/data/train_ids.csv")
    parser.add_argument('--val_idxpath', default="/home/blanka/ERLPruning/data/valid_ids.csv")
    parser.add_argument('--origpath', default="/data/blanka/DATASETS/KITTI/original")

    # For yolo training
    parser.add_argument('--weights', default= "/data/blanka/ERLPruning/runs/YOLOv4_KITTI/exp_kitti_tvt/weights/best.pt") #pretrained
    parser.add_argument('--partial_load', action='store_true')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--data', type=str, default='data/kitti.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=540, help='inference size (pixels)')
    parser.add_argument('--spndata', type=str, default='data/spndata.yaml', help='data.yaml path')
    parser.add_argument('--layers-to-skip', default=[138, 149, 160], help='Detection Layers most commonly (ex. YOLO Layers)')
    parser.add_argument('--N_features', default=7, help='nFeatures+1')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4_kitti.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/kitti.names', help='*.cfg path')
    parser.add_argument('--df_cols', default=[''])

    parser.add_argument('--test-cases', type=int, default=20000)

    opt = parser.parse_args()

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    opt.spndata = check_file(opt.spndata)  # check file

    # Create paths
    with open(opt.spndata) as f:
        pdata = yaml.load(f, Loader=yaml.FullLoader)
        state_savepath = pdata['all_state']
        label_savepath = pdata['all_label']
        df_path = pdata['allsamples']


    alphas = np.arange(0.0, 2.3, 0.1).tolist()
    alphas = [float("{:.2f}".format(x)) for x in alphas]

    glob_dims = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    dim = 0

    # Image size
    gs = 32  # grid size (max stride)
    imgsz = check_img_size(opt.img_size, gs)  # verify imgsz are gs-multiples

    # Load KITTI data
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    val_img_path = data['val_img']
    val_label_path = data['val_label']

    dataloader = \
        create_dataloader(val_img_path, val_label_path, imgsz, opt.batch_size, 32, opt, hyp=None, augment=False,
                          cache=False, rect=False)[0]
    # Load model
    model = Darknet(opt.cfg).to('cuda')
    ckpt = torch.load(opt.weights)
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)

    # Determine state size
    N_prunable_layers = prune_network(model, opt.layers_to_skip, 0, state_size_check=True)
    print(f"N_prunable_layers: {N_prunable_layers}")

    # Get metrics before pruning
    results, _, _ = test(
        opt.data,
        dataloader=dataloader,
        batch_size=opt.batch_size,
        model=model,
        opt=opt)

    prec, rec, map = results[0], results[1], results[2]
    prec_before, rec_before, map_before = prec, rec, map

    print("map", map)
    prec_before, rec_before, map_before = prec, rec, map


    network_size = len(model.module_list)
    network_param_nmb = sum([param.nelement() for param in model.parameters()])

    for test_case in range(opt.test_cases):
        # if True:

        start_time_test_case = time.time()

        num_samples = 1
        row_cnt = 0
        prev_spars = -1
        mu = 0
        sigma = 0.4
        unsorted_probs = generate_gaussian_probs(mu, sigma, len(alphas))

        model_init = Darknet(opt.cfg).to('cuda')
        ckpt = torch.load(opt.weights)
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model_init.state_dict()[k].numel() == v.numel()}
        model_init.load_state_dict(ckpt['model'], strict=False)

        state = torch.full([N_prunable_layers, opt.N_features], -1.0)
        label = torch.zeros([1, 4])  # sparsity, dmap, drec, dprec
        alpha_seq = torch.full([N_prunable_layers], 0.0)  # for real pruning

        for layer_index in range(network_size):

            start_time = time.time()

            model = Darknet(opt.cfg).to('cuda')
            model.load_state_dict(ckpt['model'], strict=False)

            # Load the dataframe containing the already existing samples
            if (os.path.exists(df_path)):
                df_allsamples = pd.read_pickle(df_path)
                sample_cnt = df_allsamples.shape[0]
            else:
                df_allsamples = pd.DataFrame(columns=opt.df_cols)
                sample_cnt = 0


            layer_param_nmb_before = sum(
                [param.nelement() for name, param in model.named_parameters() if "." + str(layer_index) + "." in name])
            param_nmb_before = sum([param.nelement() for param in model.parameters()])

            module_def = model.module_defs[layer_index]
            if module_def["type"] in ["route", "shortcut", "upsample", "maxpool", "yolo"] or layer_index in opt.layers_to_skip: # or glob_dims[layer_index] == 1: !!! BUG !!!
                continue
            layer = model.module_list[layer_index][0]  # only if convolutional

            # Generate random alpha instances
            probs = sort_gaussian_probs(unsorted_probs, len(alphas), layer_index, network_size)
            print(f"probs: {len(probs)}, alphas: {len(alphas)}, {layer_index}, {network_size}")
            alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]

            skip = random.choice([1, 2, 3, 4, 5, 6])



            if layer_index in [11, 24, 55]:
                while alpha > 0.2:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            # elif layer_index in [86, 115]:
            #    alpha = 2.2
            elif layer_index in [123, 125]:
                while alpha > 0.8:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            elif layer_index > 134 and layer_index < 155:
                while alpha > 1.3:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            elif layer_index > 154 or layer_index in [86, 115]:
                while alpha < 1.3:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]

            else:
                if layer_index > 100 and layer_index % skip == 0:
                    while alpha > 0.3:
                        alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
                else:
                    alpha = 0.0
            """

            if layer_index > 80:
                while alpha < 1.5:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            else:
                alpha = 0.0
            """


            state[row_cnt, 0] = normalize(alpha, 0, 2.2)
            alpha_seq[row_cnt] = alpha
            # alpha_seq = state[:, 0].to('cuda') ## BUG! This has to be unnormalized!!

            print(alpha_seq)
            model, parser = prune_network(model, opt.layers_to_skip, layer_index, alpha_seq, dataset_make=True)
            model.to('cuda')
            with open(f"/home/blanka/ERLPruning/sandbox/stupidmodel{layer_index}.txt", "w") as f:
                f.write(str(model))

            state[row_cnt, 1] = normalize(parser['in_ch'], 0, 1024)
            state[row_cnt, 2] = normalize(parser['out_ch'], 0, 1024)
            state[row_cnt, 3,] = normalize(parser['kernel'], 0, 3)
            state[row_cnt, 4] = normalize(parser['stride'], 0, 2)
            state[row_cnt, 5] = normalize(parser['pad'], 0, 1)
            state[row_cnt, 6] = prev_spars


            if alpha == 0.0:
                prec_after, rec_after, map_after = prec_before, rec_before, map_before
                prec_before, rec_before, map_before = prec_after, rec_after, map_after

            else:

                results, _, _ = test(
                    opt.data,
                    dataloader=dataloader,
                    batch_size=opt.batch_size,
                    model=model,
                    opt=opt)

                prec_after, rec_after, map_after = results[0], results[1], results[2]
                prec_before, rec_before, map_before = prec_after, rec_after, map_after

            print("prec_after, rec_after, map_after", prec_after, rec_after, map_after)

            param_nmb_after = sum([param.nelement() for param in model.parameters()])
            # print("remaining params ratio", param_nmb_after/param_nmb_before) # BUG!
            print(param_nmb_after, network_param_nmb)
            print(param_nmb_after, param_nmb_before)

            print("remaining params ratio", param_nmb_after / network_param_nmb)
            layer_param_nmb_after = sum(
                [param.nelement() for name, param in model.named_parameters() if "." + str(layer_index) + "." in name])

            pruned_perc = round((1 - param_nmb_after / network_param_nmb) * 100, 4)
            mAPa = round(map_after, 10)
            spars = normalize(float(pruned_perc), 0, 100)  # percent of pruned params from the whole network
            dmap = normalize(1.0 - (float(mAPa) / float(map)), 0, 1)
            drec = normalize(1.0 - (float(rec_after) / float(rec)), 0, 1)
            dprec = normalize(1.0 - (float(prec_after) / float(prec)), 0, 1)

            label[0, 0], label[0, 1], label[0, 2], label[0, 3] = spars, dmap, drec, dprec

            prev_spars = spars
            row_cnt += 1

            elapsed_time = time.time() - start_time
            print("Runtime of one layer pruning: ", elapsed_time)

            if layer_index == 159:
                prec_before, rec_before, map_before = prec, rec, map

            # SAVE

            print(f"Sample_idx, layer_idx, alpha: {sample_cnt}, {layer_index}, {alpha}")
            print("allsamples_shape", df_allsamples.shape)

            state_tosave = state.reshape(N_prunable_layers * opt.N_features).unsqueeze(dim=0)
            df_state = pd.DataFrame([str(state_tosave)], columns=opt.df_cols)

            if (df_allsamples == str(state_tosave)).all(1).any():
                print("This state already exists in the dataset.")
                continue
            else:
                if os.path.exists(os.path.join(state_savepath, str(sample_cnt) + ".txt")):
                    print(f"Sample {sample_cnt} already exists!")
                    break
                elif os.path.exists(os.path.join(label_savepath, str(sample_cnt) + ".txt")):
                    print(f"Label {sample_cnt} already exists!")
                    break
                else:
                    np.savetxt(os.path.join(state_savepath, str(sample_cnt) + ".txt"), state.numpy())
                    np.savetxt(os.path.join(label_savepath, str(sample_cnt) + ".txt"), label.numpy())

                if df_allsamples.empty:
                    df_allsamples = df_state
                else:
                    df_allsamples = pd.concat([df_allsamples, df_state], axis=0)
                df_allsamples.to_pickle(df_path)
                sample_cnt += 1

        print("Runtime of one testcase: ", time.time() - start_time_test_case)


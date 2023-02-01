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

    if global_print: print("len(indexes) ", len(indices))

    return indices.squeeze(dim=1).tolist()

def prune_network(network, yolo_layers, layer_to_prune, alpha_seq, device='cuda', dataset_make=False):


    # if dim == 0 --> prune output channels
    # if dim == 1 --> prune input channels
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    layer_cnt = 0
    network_size = len(network.module_list)
    dims = [0] * 1
    alphas = []
    prev_indexes = []
    output_filters = []
    old_output_filters = []
    index_container = []
    index_lens = []
    yolo_flag = False

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

    save_path = "/home/blanka/YOLOv4_Pruning/sandbox/" + "pruning_develop_data.txt"
    # with open(save_path, 'w') as f:
    if True:
        for i in range(network_size):

            start_time_forloop = time.time()
            print(i)

            sequential_size = len(network.module_list[i])
            module_def = network.module_defs[i]
            if global_filewrite: f.write("\nmodule_def " + str(module_def["type"]) + str(module_def) + "\n")

            if i in yolo_layers:
                print("Skipped, as it is a YOLO Layer.")
                continue

            if module_def["type"] == "convolutional":

                    layer = network.module_list[i][0] # if j == 0: nn.Conv2d instance (the rest is BacthNorm and activation)

                    if global_filewrite: f.write("Pruning layer " + str(i) + " " + str(layer) + "\n")

                    if isinstance(layer, nn.Conv2d) and glob_dims[i] == 0:

                        #alpha = np.around(alpha_seq[layer_cnt].item(), 1)
                        alpha = float(torch.round(alpha_seq[layer_cnt]).item()) ## BUG !!!!
                        #alpha = 0.2
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

                        print(f"Error! The layer is not an instance of nn.Conv2d! \n{layer}")


    if dataset_make:
        return network, parser

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


    opt = parser.parse_args()

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

    if opt.weights:
        if opt.weights.endswith(".pt"):
            ckpt = torch.load(opt.weights)

            if opt.partial_load:
                for k, v in ckpt['model'].items():
                    if model.state_dict()[k].numel() != v.numel():
                        print(f"Weights are not loaded to layer {k}")

                state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(state_dict, strict=False)
                print('Transferred %g/%g items from %s' % (
                len(state_dict), len(model.state_dict()), opt.weights))  # report
            else:
                if 'arch' in ckpt:
                    print("arch in ckpt")
                    model = ckpt['arch']
                else:
                    print(f"Error: ckpt has no key 'arch'.")

        elif opt.weights.endswith('.weights'):
            model.load_darknet_weights(opt.weights)

    # Get metrics before pruning
    results, _, _ = test(
        opt.data,
        dataloader=dataloader,
        batch_size=opt.batch_size,
        model=model,
        opt=opt)

    map_before = results[2]

        ##
    network_size = len(model.module_list)
    network_param_nmb = sum([param.nelement() for param in model.parameters()])

    yolo_layers = [138, 148, 149, 160]
    alpha_seq = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0.0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]).to("cuda")  # RL
    model, parser = prune_network(model, yolo_layers, layer_to_prune=0, alpha_seq=alpha_seq, dataset_make=True)

    results, _, _ = test(
        opt.data,
        dataloader=dataloader,
        batch_size=opt.batch_size,
        model=model,
        opt=opt)

    prec_after, rec_after, map_after = results[0], results[1], results[2]

    param_nmb_after = sum([param.nelement() for param in model.parameters()])
    # print("remaining params ratio", param_nmb_after / network_param_nmb)

    pruned_perc = round((1 - param_nmb_after / network_param_nmb) * 100, 4)
    map_after = round(map_after, 10)
    dperf = 1.0 - (float(map_after) / float(map_before))
    spars_norm = normalize(float(pruned_perc), 0, 100)  # percent of pruned params from the whole network
    dperf_norm = normalize(1.0 - (float(map_after) / float(map_before)), 0, 1)

    print(f"mAP before, after: {map_before}, {map_after}")
    print(f"param nmb before, after: {network_param_nmb}, {param_nmb_after}")
    print(f"dperf, sparsity: {dperf}, {pruned_perc}")
    print(f"norm dperf, sparsity: {dperf_norm}, {spars_norm}")


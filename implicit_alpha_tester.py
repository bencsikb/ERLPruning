import torch
import argparse
import torch.nn as nn
import math
from numpy import linalg as LA
import random
import numpy as np
import itertools
import time
import pandas as pd
import yaml

#from prune_for_error import prune_network, normalize
from auto_pruning import prune_network, normalize
from models.models import *
from utils.layers import *
from test import *
from utils.datasets import *
from utils.general import *

def get_results(alpha_seq, map_before=None, layer_index=43):
    yolo_layers = [138, 148, 149, 160]
    glob_dims = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    dim = 0

    # Create paths
    with open(opt.spndata) as f:
        pdata = yaml.load(f, Loader=yaml.FullLoader)

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

    # Get metrics before pruning
    if map_before is None:
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

    model, parser = prune_network(model, yolo_layers, layer_index, alpha_seq, dataset_make=True)
    model.to("cuda")

    # Calculate metrics after
    results, _, _ = test(
        opt.data,
        dataloader=dataloader,
        batch_size=opt.batch_size,
        model=model,
        opt=opt)

    prec_after, rec_after, map_after = results[0], results[1], results[2]

    param_nmb_after = sum([param.nelement() for param in model.parameters()])
    #print("remaining params ratio", param_nmb_after / network_param_nmb)

    pruned_perc = round((1 - param_nmb_after / network_param_nmb) * 100, 4)
    map_after = round(map_after, 10)
    dperf = 1.0 - (float(map_after) / float(map_before))
    spars_norm = normalize(float(pruned_perc), 0, 100)  # percent of pruned params from the whole network
    dperf_norm = normalize(1.0 - (float(map_after) / float(map_before)), 0, 1)


    print(f"mAP before, after: {map_before}, {map_after}")
    print(f"param nmb before, after: {network_param_nmb}, {param_nmb_after}")
    print(f"dperf, sparsity: {dperf}, {pruned_perc}")
    print(f"norm dperf, sparsity: {dperf_norm}, {spars_norm}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default="weights/yolov4_kitti.weights", help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str,
                        default="/data/blanka/ERLPruning/runs/YOLOv4_KITTI/exp_kitti_tvt/weights/best.pt",
                        help='model.pt path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default="/data/blanka/ERLPruning/runs/YOLOv4_PascalVoc/exp_pascalvoc_scratch_2/weights/last.pt", help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/kitti.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=540, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', default='False', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4_kitti.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/kitti.names', help='*.cfg path')
    # parser.add_argument('--save-path', type=str, default='sandbox/DatasetForPruning')
    parser.add_argument('--spndata', type=str, default='data/spndata.yaml', help='data.yaml path')
    parser.add_argument('--state_size', default=[44, 7], help='nPrunableLayers, nFeatures+1')
    # parser.add_argument('--df_cols', default=['alpha', 'in_channel', 'out_channel', 'kernel', 'stride', 'pad', 'spars'])
    parser.add_argument('--df_cols', default=[''])
    parser.add_argument('--test-cases', type=int, default=5000)

    opt = parser.parse_args()

    alpha_seq = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.9, 0, 0, 2.2, 0.1, 2.2, 0.3, 1.2, 0.1, 0.1, 0, 0, 0, 0, 0.1, 0.5, 0.1, 2.1, 1.8, 2.2]).to("cuda")  # RL
    #alpha_seq = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.2,0,0,0,0,0,0,2.2,0,0.1,0,0,0,0,0,0.2,0,2.2,2.2,2.2]) # handcrafted1
    #alpha_seq = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0,2.2,0,0,0,0,0,0,2.2,0,0.2,0.2,0,0,0.1,0,0,0.1,2.2,2.2,0]) # handcrafted10
    #alpha_seq = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #alpha_seq = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0.0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]).to("cuda")  # RL

    get_results(alpha_seq, map_before=0.726)

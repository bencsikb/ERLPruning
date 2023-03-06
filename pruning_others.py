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
import torch_pruning as tp

from auto_pruning import prune_network, normalize

""" torchvision 0.14.1 required """
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import models


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
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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

    model = models.alexnet(pretrained=True).to(opt.device)

    #weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    #model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9).to(opt.device)
    #model.eval()
    #out = model(torch.randn(1, 3, 224, 224).to(opt.device))
    #print(out)

    #for i, (name, layer) in enumerate(model.named_modules()):
    #    if i == 0:
    #        print(layer)

    # Determine state size
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224).to(opt.device))


    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):  # and glob_dims[i] == 0: # only if testing old 44 long alpha_seqs
            print(layer)


            pruning_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=[1,3])

            if DG.check_pruning_group(pruning_group):
                pruning_group.exec()

            print(layer)


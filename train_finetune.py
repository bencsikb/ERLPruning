import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
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

#from utils.data_converter import get_files2split, split_kitti
#import test  # import test.py to get mAP after each epoch
from test import *
from models.models import *
#from prune import *
#from prune_develop import *
from prune_for_error import *
from utils.datasets import create_dataloader
from utils.general import (
    check_img_size, torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors,
    labels_to_image_weights, compute_loss, plot_images, fitness, strip_optimizer, plot_results,
    get_latest_run, check_git_status, check_file, increment_dir, print_mutation, plot_evolution)
from utils.google_utils import attempt_download
from utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts

def train(model, optimizer, scheduler, hyp, opt, device, dataset, dataloader, dataloader_val, accumulate, best_fitness, tb_writer=None):

    print(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
    wdir = str(log_dir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = str(log_dir / 'results.txt')
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    cuda = device.type != 'cpu'
    #init_seeds(2 + rank)
    init_seeds(42)

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=(opt.local_rank))

    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    results_train = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    if rank in [0, -1]:
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)


    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        
        if epoch > 0:
            with open('sandbox/finetune_nextepoch.txt', "w") as f:
                f.write(str(model))

        # Update image weights (optional)
        if dataset.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                 k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # print("Targets in train ", targets)


            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])


            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Autocast
            with amp.autocast(enabled=cuda):
                # Forward
                pred = model(imgs)

                # Loss
                                
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                # if not torch.isfinite(loss):
                #     print('WARNING: non-finite loss, ending training ', loss_items)
                #     return results

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()
        lr_print = 'Learning rate (scheduler) at this epoch is: %0.9f' % scheduler.get_lr()[0]
        print(lr_print)

        # DDP process 0 or single-GPU
        # Evaluate
        if rank in [-1, 0]:
            # mAP

            if ema is not None:
                ema.update_attr(model)
            final_epoch = epoch + 1 == epochs


            # Evaluate on train dataset
            #TODO if not opt.notest or final_epoch:  # Calculate mAP
            if False:
                if epoch % opt.train_eval_period == 0:
                    results_train, maps_train, _ = test(opt.data,
                                                imgsz=imgsz_test,
                                                save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                                model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                single_cls=opt.single_cls,
                                                dataloader=dataloader,
                                                training=True)


                    results, maps, times = test(opt.data,
                                                     imgsz=imgsz_test,
                                                     save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                                     model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                     single_cls=opt.single_cls,
                                                    dataloader=dataloader_val,
                                                    training=True)



            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results_train +'%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))


            # Tensorboard
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'train/precision', 'train/recall', 'train/mAP_0.5',
                        'val/precision', 'val/recall', 'val/mAP_0.5', 'val/mAP_0.5:0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                train_prec, train_rec, train_map, _, _, _, _ = results_train
                for x, tag in zip(list(mloss[:-1]) + [train_prec, train_rec, train_map] + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi


            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                print("epoch being saved", epoch)
                with open(results_file, 'r') as f:  # create checkpoint

                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                            'arch' : model,
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                torch.save(ckpt, wdir + str(epoch) +'.pt')
                with open('sandbox/finetune_endtrain.txt', "w") as f:
                    f.write(str(model))
                if epoch >= (epochs - 5):
                    torch.save(ckpt, last.replace('.pt', '_{:03d}.pt'.format(epoch)))
                if (best_fitness == fi) and not final_epoch:
                    torch.save(ckpt, best)

                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload
        # Finish
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default='/data/blanka/ERLPruning/runs/YOLOv4_KITTI/exp_finetune/weights/last.pt')
    parser.add_argument('--weights', default="sandbox/only_models/pruned_RL.pt")
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--cfg', type=str, default='cfg/models/yolov4_kitti.cfg', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/kitti.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.finetune.yaml', help='hyperparameters path')
    parser.add_argument('--logdir', type=str, default="/data/blanka/ERLPruning/runs/YOLOv4_KITTI/", help='logging directory')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train_eval_period', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[540, 540], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='/data/blanka/ERLPruning/runs/YOLOv4_KITTI', help='save to project/name')
    parser.add_argument('--name', default='exp_finetune', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--test_case', default="exp_finetune2")

    opt = parser.parse_args()


    # Split KITTI dataset
    #get_files2split(opt.labelpath, opt.savepath, 0.7, opt.batch_size)
    #split_kitti(opt.origpath, opt.train_idxpath, opt.val_idxpath)

    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = select_device(opt.device, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    opt.global_rank = -1

    # DDP mode
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        opt.world_size = dist.get_world_size()
        opt.global_rank = dist.get_rank()
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Writer
    #tb_writer = None
    #if opt.global_rank in [-1, 0]:
    #    print('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
        # tb_writer = SummaryWriter(log_dir=increment_dir(Path(opt.logdir) / 'exp', opt.name))  # runs/exp

    # tb_writer = SummaryWriter(log_dir=Path(opt.logdir + 'edfjxp_pruned_L-' + str(layer_to_prune) + "_A-" + str(alpha)))  # runs/exp
    tb_writer = SummaryWriter(log_dir=Path(opt.logdir + opt.test_case))  # runs/exp

    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
    results_file = str(log_dir / 'results.txt')
    opt_file = str(log_dir / 'logs.txt')
    with open(opt_file, 'w') as f:
        f.write(str(opt))

    # Hyperparameters

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Dataset

    # Image sizes
    gs = 32  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    train_img_path = data_dict['train_img']
    val_img_path = data_dict['val_img']

    train_label_path = data_dict['train_label']
    val_label_path = data_dict['val_label']

    dataloader, dataset = create_dataloader(train_img_path, train_label_path, imgsz, opt.batch_size, gs, opt, hyp=hyp,
                                            augment=True,
                                            cache=opt.cache_images, rect=opt.rect, local_rank=opt.global_rank,
                                            world_size=opt.world_size)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    dataloader_val, dataset_val = create_dataloader(val_img_path, val_label_path, imgsz, 1, gs, opt.single_cls,
                                                    hyp=hyp,
                                                    augment=False,
                                                    cache=opt.cache_images, rect=opt.rect, local_rank=opt.global_rank,
                                                    world_size=opt.world_size)

    # Model

    if opt.pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Darknet(opt.cfg).to(device)  # create
        state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(state_dict, strict=False)
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = torch.load(opt.weights, map_location=device)
        #model = ckpt['arch']
        with open('sandbox/finetune_init.txt', "w") as f:
            f.write(str(model))


    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= opt.total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'Conv2d.weight' in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / opt.epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.0002, last_epoch=-1)
    # lr_print = 'Learning rate at this epoch is: %0.9f' % scheduler.get_lr()[0]
    # print(lr_print)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Resume
    start_epoch, best_fitness = 0, 0.0

    if opt.resume:
        print("Training is resumed")

        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        print(f"ckpt epoch {ckpt['epoch']}")
        start_epoch = ckpt['epoch'] + 1
        print("epoch in loaded ckpt", ckpt['epoch'])
        if opt.epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], opt.epochs))
            opt.epochs += ckpt['epoch']  # finetune additional epochs

        # Learning rate
        scheduler.load_state_dict(ckpt['scheduler'])
        scheduler.step()
        lr_print = 'Learning rate at this epoch is: %0.9f' % scheduler.get_lr()[0]
        print(lr_print)

        del ckpt

    # Model parameters

    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names



    # Train

    train(model, optimizer, scheduler, hyp, opt, device, dataset, dataloader, dataloader_val, accumulate, best_fitness,
          tb_writer)



"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

import argparse
import os
import sys
import time
import math
import datetime
from typing import Iterable

import torch
from torch.cuda.amp import GradScaler
import torch.optim.lr_scheduler as lr_schedulers
import torch.amp 
import wandb


from src.zoo import rtdetr_criterion
from src.data.coco.coco_eval import CocoEvaluator
from src.data.coco.coco_utils import get_coco_api_from_dataset
from src.misc import MetricLogger, SmoothedValue, reduce_dict
from src.optim.ema import ModelEMA
from src.nn.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from src.nn.rtdetr.utils import *
import src.misc.dist_utils as dist_utils


def fit(model, 
        weight_path, 
        optimizer, 
        save_dir,
        train_dataloader, 
        val_dataloader,
        criterion=None,
        epoch=73,
        use_amp=True,
        use_ema=True,
        use_wandb=False):

    if criterion == None:
        criterion = rtdetr_criterion()

    scaler = GradScaler() if use_amp == True else None
    ema_model = ModelEMA(model, decay=0.9999, warmups=2000) if use_ema == True else None
    lr_scheduler = lr_schedulers.MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=0.1) 

    last_epoch = 0
    if weight_path != None:
        last_epoch = load_tuning_state(weight_path, model, ema_model)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ema_model.to(device) if use_ema == True else None
    criterion.to(device)  
    
    #dist wrap modeln loader must do after model.to(device)
    if dist_utils.is_dist_available_and_initialized():
        print("Distributed training is enabled, wrapping model and dataloaders")
        train_dataloader = dist_utils.warp_loader(train_dataloader)
        val_dataloader = dist_utils.warp_loader(val_dataloader)
        model = dist_utils.warp_model(model, find_unused_parameters=False, sync_bn=True)

    
    print("Start training")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    start_time = time.time()
    

    for epoch in range(last_epoch + 1, epoch):
        sys.stdout = Tee(os.path.join(save_dir, f'{epoch}.txt'))

        # set dataloader epoch parameter
        train_dataloader.sampler.set_epoch(epoch) if dist_utils.is_dist_available_and_initialized() else train_dataloader.set_epoch(epoch)
        
        train_one_epoch(model, criterion, train_dataloader, optimizer, device, epoch, max_norm=0.1, print_freq=100, ema=ema_model, scaler=scaler, use_wandb=use_wandb)

        lr_scheduler.step()

        dist_utils.save_on_master(state_dict(epoch, model, ema_model), os.path.join(save_dir, f'{epoch}.pth'))

        # The val function during training is always use_ema=False flag to skip the logic of fetching ema files
        module = ema_model.module if use_ema == True else model
        test_stats, coco_evaluator = val(model=module, weight_path=None, criterion=criterion, val_dataloader=val_dataloader, use_wandb=use_wandb, use_amp=use_amp, use_ema=False)

        sys.stdout.close()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    print_freq: int,
                    max_norm: float = 0,
                    ema=None, 
                    scaler=None,
                    use_wandb=False):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(data_loader, header=f'Epoch: [{epoch}]', print_freq=print_freq)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for samples, targets in metric_logger.log_every():
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #amp
        if scaler != None:
            with torch.autocast(device_type=device.type, cache_enabled=True, enabled=device.type == 'cuda'):
                outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            # weight_dict = criterion.weight_dict
            # loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            # weight_dict = criterion.weight_dict
            # loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema != None:
            ema.update(model)

        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_value = sum(loss_dict_reduced.values())
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items()}
        total_loss_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = total_loss_reduced_scaled.item()

        # metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if use_wandb and dist_utils.is_main_process():
            log_data = {"train/" + k: v.item() for k, v in loss_dict_reduced.items()}
            log_data["train/total_loss"] = loss_value
            log_data["train/lr"] = optimizer.param_groups[0]["lr"]
            log_data["epoch"] = epoch
            wandb.log(log_data)


#TODO This function too complex and slow because it from original repository, need to refactor
@torch.no_grad()
def val(model, weight_path, val_dataloader, criterion=None, use_amp=True, use_ema=True, use_wandb=False):
    if criterion == None:
        criterion = rtdetr_criterion()

    if weight_path != None:
        state = torch.hub.load_state_dict_from_url(weight_path, map_location='cpu') if 'http' in weight_path else torch.load(weight_path, map_location='cpu')
        if use_ema == True:
            model.load_state_dict(state['ema']['module'], strict=False)
        else:
            model.load_state_dict(state['model'], strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    if dist_utils.is_dist_available_and_initialized():
        val_dataloader = dist_utils.warp_loader(val_dataloader)
        model = dist_utils.warp_model(model, find_unused_parameters=False, sync_bn=True)
    
    model.eval()
    criterion.eval()

    base_ds = get_coco_api_from_dataset(val_dataloader.dataset)

    postprocessor = RTDETRPostProcessor(num_top_queries=300, remap_mscoco_category=val_dataloader.dataset.remap_mscoco_category)
    # coco_evaluator = CocoEvaluator(base_ds, ['bbox'])
    # iou_types = coco_evaluator.iou_types
    iou_types = ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    metric_logger = MetricLogger(val_dataloader, header='Test:',)

    # print(f"[DEBUG] postprocessor: {postprocessor}")
    # print(f"Coco evaluator: {coco_evaluator}")
    # print(f"metric logger: {metric_logger}")

    for samples, targets in metric_logger.log_every():
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=use_amp == True and device.type == 'cuda'):
            outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessor(outputs, orig_target_sizes)
        # print(f"[DEBUG] results: {results}")

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # # convert tensor ke cpu
        # res = {target['image_id'].item(): {k: v.cpu() for k, v in output.items()} 
        #        for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}

    if use_wandb and dist_utils.is_main_process():
        log_data = {}
        if coco_evaluator is not None:
            if 'bbox' in iou_types:
                bbox_stats = coco_evaluator.coco_eval['bbox'].stats
                stats['coco_eval_bbox'] = bbox_stats.tolist() 
                log_data.update({
                    "val/bbox_AP": bbox_stats[0],           # AP @ IoU=.50:.05:.95 | area=all
                    "val/bbox_AP50": bbox_stats[1],         # AP @ IoU=.50        | area=all
                    "val/bbox_AP75": bbox_stats[2],         # AP @ IoU=.75        | area=all
                    "val/bbox_AP_small": bbox_stats[3],     # AP @ IoU=.50:.05:.95 | area=small
                    "val/bbox_AP_medium": bbox_stats[4],    # AP @ IoU=.50:.05:.95 | area=medium
                    "val/bbox_AP_large": bbox_stats[5],     # AP @ IoU=.50:.05:.95 | area=large
                    "val/bbox_AR_1": bbox_stats[6],         # AR @ IoU=.50:.05:.95 | area=all | maxDets=1
                    "val/bbox_AR_10": bbox_stats[7],        # AR @ IoU=.50:.05:.95 | area=all | maxDets=10
                    "val/bbox_AR_100": bbox_stats[8],       # AR @ IoU=.50:.05:.95 | area=all | maxDets=100
                    "val/bbox_AR_small": bbox_stats[9],     # AR @ IoU=.50:.05:.95 | area=small
                    "val/bbox_AR_medium": bbox_stats[10],   # AR @ IoU=.50:.05:.95 | area=medium
                    "val/bbox_AR_large": bbox_stats[11],    # AR @ IoU=.50:.05:.95 | area=large
                })
            if 'segm' in iou_types:
                segm_stats = coco_evaluator.coco_eval['segm'].stats
                stats['coco_eval_masks'] = segm_stats.tolist()                
                log_data.update({
                    "val/segm_AP": segm_stats[0],
                    "val/segm_AP50": segm_stats[1],
                    "val/segm_AP75": segm_stats[2],
                    "val/segm_AP_small": segm_stats[3],
                    "val/segm_AP_medium": segm_stats[4],
                    "val/segm_AP_large": segm_stats[5],
                    "val/segm_AR_1": segm_stats[6],
                    "val/segm_AR_10": segm_stats[7],
                    "val/segm_AR_100": segm_stats[8],
                    "val/segm_AR_small": segm_stats[9],
                    "val/segm_AR_medium": segm_stats[10],
                    "val/segm_AR_large": segm_stats[11],
                })
            
        if log_data:
            wandb.log(log_data)
    else:
        if coco_evaluator is not None:
            if 'bbox' in iou_types:
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in iou_types:
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log_file = open(path, 'w')
        self.file = log_file
        self.stdout = sys.stdout

    def write(self, obj):
        self.file.write(obj)
        self.file.flush()
        self.stdout.write(obj)
        self.stdout.flush()


    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        sys.stdout = sys.__stdout__  # Restore original stdout
        self.file.close()
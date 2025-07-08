"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import os
import random
import numpy as np 
import atexit

import torch
import torch.nn as nn 
import torch.distributed
import torch.backends.cudnn

from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data import DistributedSampler
from ..data.dataloader import DataLoader


def init_distributed(print_rank: int=0, print_method: str='builtin', seed: int=None, ):
    """
    env setup
    args:
        print_rank, 
        print_method, (builtin, rich)
        seed, 
    """
    try:
        # https://pytorch.org/docs/stable/elastic/run.html
        RANK = int(os.getenv('RANK', -1))
        LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
        WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
        
        # torch.distributed.init_process_group(backend=backend, init_method='env://')
        torch.distributed.init_process_group(init_method='env://')
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        enabled_dist = True
        print('Initialized distributed mode...')

    except:
        enabled_dist = False
        print('Not init distributed mode.')

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)

    return enabled_dist


def setup_print(is_main, method='builtin'):
    """This function disables printing when not in master process
    """
    import builtins as __builtin__

    if method == 'builtin':
        builtin_print = __builtin__.print

    elif method == 'rich':
        import rich 
        builtin_print = rich.print

    else:
        raise AttributeError('')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    # print("is_dist_available_and_initialized")
    print("torch.distributed.is_available():", torch.distributed.is_available())
    print("torch.distributed.is_initialized():", torch.distributed.is_initialized())
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@atexit.register
def cleanup():
    """cleanup distributed environment
    """
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()

    
def is_main_process():
    return get_rank() == 0


def save_on_master(obj, file_path):
    if is_main_process():
        save_dir = os.path.dirname(file_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(obj, file_path)



def warp_model(
    model: torch.nn.Module, 
    sync_bn: bool=False, 
    dist_mode: str='ddp', 
    find_unused_parameters: bool=False, 
    compile: bool=False, 
    compile_mode: str='reduce-overhead', 
    **kwargs
):
    if is_dist_available_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model 
        if dist_mode == 'dp':
            model = DP(model, device_ids=[rank], output_device=rank)
        elif dist_mode == 'ddp':
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
        else:
            raise AttributeError('')

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model

def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):        
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset, 
                            loader.batch_size, 
                            sampler=sampler, 
                            drop_last=loader.drop_last, 
                            collate_fn=loader.collate_fn, 
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers, )
    return loader



def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args 
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data
    
    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size
        
        return {k: v for k, v in zip(keys, values)}
        

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list

    
def sync_time():
    """sync_time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()


def setup_seed(seed: int, deterministic=False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # memory will be large when setting deterministic to True
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


# for torch.compile
def check_compile():
    import torch
    import warnings
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    return gpu_ok

def is_compile(model):
    import torch._dynamo
    return type(model) in (torch._dynamo.OptimizedModule, )

def de_complie(model):
    return model._orig_mod if is_compile(model) else model

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from typing import List, Optional

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        ## type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
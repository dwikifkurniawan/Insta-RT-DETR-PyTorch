"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

import torch
from src.data.coco.coco_dataset import CocoDetection_share_memory
from src.data import transforms as T
import torchvision.transforms.v2 as T2

import random
import numpy as np
from torchvision.tv_tensors import Mask, BoundingBoxes
from PIL import Image


class CopyPaste(T2.Transform):
    def __init__(self, dataset, p=0.5):
        super().__init__()
        self.dataset = dataset
        self.p = p

    def forward(self, image, target):
        if random.random() > self.p:
            return image, target

        source_idx = random.randint(0, len(self.dataset) - 1)
        source_image, source_target = self.dataset.load_item(source_idx)

        resize_transform = T.Resize([640, 640])
        image, target = resize_transform(image, target)
        source_image, source_target = resize_transform(source_image, source_target)

        num_objects = len(source_target["labels"])
        if num_objects == 0:
            return image, target
        
        num_to_paste = random.randint(1, num_objects)
        indices_to_paste = random.sample(range(num_objects), k=num_to_paste)

        pasted_image_np = np.array(image)
        source_image_np = np.array(source_image)

        for i in indices_to_paste:
            source_mask = source_target["masks"][i].numpy().astype(bool)
            pasted_image_np[source_mask] = source_image_np[source_mask]
        
        pasted_image = Image.fromarray(pasted_image_np)

        pasted_boxes_data = source_target["boxes"][indices_to_paste].data
        pasted_labels = source_target["labels"][indices_to_paste]
        pasted_masks_data = source_target["masks"][indices_to_paste].data

        device = target["boxes"].device
        pasted_boxes_data = pasted_boxes_data.to(device)
        pasted_labels = pasted_labels.to(device)
        pasted_masks_data = pasted_masks_data.to(device)

        new_boxes_data = torch.cat([target["boxes"].data, pasted_boxes_data])
        new_labels = torch.cat([target["labels"], pasted_labels])
        new_masks_data = torch.cat([target["masks"].data, pasted_masks_data])

        h, w = pasted_image.height, pasted_image.width
        target["boxes"] = BoundingBoxes(
            new_boxes_data,
            format=target["boxes"].format,
            canvas_size=(h, w)
        )
        target["labels"] = new_labels
        target["masks"] = Mask(new_masks_data)
        
        return pasted_image, target


def coco_train_dataset(
        img_folder="./dataset/coco/train2017/",
        ann_file="./dataset/coco/annotations/instances_train2017.json",
        range_num=None,
        dataset_class=CocoDetection_share_memory,
        use_copy_paste=False,
        **kwargs):
    
    transforms_list = []
    if use_copy_paste:
        print("Using Copy-Paste augmentation")
        temp_dataset = dataset_class(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=None, 
            return_masks=True,
            remap_mscoco_category=True,
            **kwargs
        )
        transforms_list.append(CopyPaste(dataset=temp_dataset, p=0.5))
    
    transforms_list.extend([
        T.RandomPhotometricDistort(p=0.5), 
        T.RandomZoomOut(fill=0), 
        T.RandomIoUCrop(p=0.8),
        T.SanitizeBoundingBoxes(min_size=1),
        T.RandomHorizontalFlip(),
        T.Resize(size=[640, 640]),
        T.SanitizeBoundingBoxes(min_size=1),
        T.ConvertPILImage(dtype='float32', scale=True),
        T.ConvertBoxes(fmt='cxcywh', normalize=True)
    ])
    
    train_dataset = dataset_class(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms = T.Compose(transforms_list),
        return_masks=True,
        remap_mscoco_category=True, 
        **kwargs)
    
    if range_num != None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(range_num))

    return train_dataset


def coco_val_dataset(
        img_folder="./dataset/coco/val2017/",
        ann_file="./dataset/coco/annotations/instances_val2017.json",
        range_num=None,
        dataset_class=CocoDetection_share_memory,
        **kwargs):
    
    val_dataset = dataset_class(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=T.Compose([T.Resize(size=[640, 640]), 
                              T.ConvertPILImage(dtype='float32', scale=True)]),
        return_masks=True,
        remap_mscoco_category=True,
        **kwargs)
    
    if range_num != None:
        val_dataset = torch.utils.data.Subset(val_dataset, range(range_num))

    return val_dataset
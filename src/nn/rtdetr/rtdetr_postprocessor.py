"""
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

This file has been revised to merge the original RT-DETR detection
post-processing logic with added support for instance segmentation masks.
The mask output shape has been specifically adjusted to [N, 1, H, W]
to conform to a fixed, legacy COCO evaluation script.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision


class RTDETRPostProcessor(nn.Module):
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False,
        mask_threshold=0.5
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 
        self.mask_threshold = mask_threshold

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
    # def forward(self, outputs, targets: list):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        pred_masks = outputs.get('pred_masks', None)

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        batch_size = logits.shape[0]

        # Ini detection dari RT-DETR
        scaled_boxes = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy') # convert boxes ke xyxy
        scaled_boxes *= orig_target_sizes.repeat(1, 2).unsqueeze(1) # scale boxes ke ukuran asli

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores_flat = scores.flatten(1)
            top_scores, top_indices = torch.topk(scores_flat, self.num_top_queries, dim=-1)
            top_labels = top_indices % self.num_classes
            query_indices = top_indices // self.num_classes
            batch_indices = torch.arange(len(logits), device=logits.device).unsqueeze(1)
            selected_boxes = scaled_boxes[batch_indices, query_indices]
        else: 
            scores_softmax = F.softmax(logits)[:, :, :-1]
            top_scores, top_labels = scores_softmax.max(dim=-1)
            selected_boxes = scaled_boxes
            if top_scores.shape[1] > self.num_top_queries:
                top_scores, query_indices = torch.topk(top_scores, self.num_top_queries, dim=-1)
                top_labels = torch.gather(top_labels, dim=1, index=query_indices)
                selected_boxes = torch.gather(selected_boxes, dim=1, index=query_indices.unsqueeze(-1).tile(1, 1, 4))
            else:
                query_indices = torch.arange(top_scores.shape[1], device=logits.device).unsqueeze(0).expand(batch_size, -1)
        
        # segmentation
        results = []
        for i in range(batch_size):
            result = {
                'scores': top_scores[i],
                'labels': top_labels[i],
                'boxes': selected_boxes[i],
            }

            # proses mask
            if pred_masks is not None and len(query_indices[i]) > 0:
                selected_masks_raw = pred_masks[i, query_indices[i]]
                height, width = orig_target_sizes[i].int().tolist()
                # height, width = targets[i]['orig_size'].int().tolist()
                
                # Apply sigmoid and upsample to target size
                # Shape: [N, H_mask, W_mask] -> [N, 1, H_target, W_target]
                mask_logits = selected_masks_raw.sigmoid()
                
                # Add channel dimension for interpolation
                mask_logits = mask_logits.unsqueeze(1)  # [N, 1, H_mask, W_mask]
                
                # Interpolate to target size
                upsampled_masks = F.interpolate(
                    mask_logits,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )
                
                binary_masks = upsampled_masks > self.mask_threshold                
                result['masks'] = binary_masks
                
            elif pred_masks is not None:
                # No valid detections but masks are expected
                height, width = orig_target_sizes[i].int().tolist()
                # height, width = targets[i]['orig_size'].int().tolist()
                result['masks'] = torch.empty(0, 1, height, width, device=logits.device, dtype=torch.bool)

            if self.remap_mscoco_category:
                from ...data.coco import mscoco_label2category
                result['labels'] = torch.tensor(
                    [mscoco_label2category[int(lbl)] for lbl in result['labels']],
                    device=result['labels'].device, 
                    dtype=result['labels'].dtype
                )

            results.append(result)
        
        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
"""
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision


def mod(a, b):
    out = a - a // b * b
    return out


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
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes, pred_masks = outputs['pred_logits'], outputs['pred_boxes'], outputs.get('pred_masks', None)
        
        if self.use_focal_loss:
            scores = logits.sigmoid()
            scores_flat = scores.flatten(1)
            num_top = min(self.num_top_queries, scores_flat.shape[1])
            top_scores, top_indices = torch.topk(scores_flat, num_top, dim=-1)
            labels = top_indices % self.num_classes
            query_indices = top_indices // self.num_classes
        else: 
            scores_softmax, labels = logits.softmax(-1)[..., :-1].max(-1)
            top_scores, query_indices = scores_softmax.topk(self.num_top_queries, dim=-1)
            labels = torch.gather(labels, 1, query_indices)

        batch_indices = torch.arange(len(logits), device=logits.device).unsqueeze(1)
        selected_boxes = boxes[batch_indices, query_indices]

        # Scale boxes (bbox conversion directly to img scale)
        scaled_boxes = torchvision.ops.box_convert(selected_boxes, 'cxcywh', 'xyxy')
        img_h, img_w = orig_target_sizes[:, 0:1], orig_target_sizes[:, 1:2]
        scale_fct = torch.cat([img_w, img_h, img_w, img_h], dim=1).unsqueeze(1)
        scaled_boxes *= scale_fct

        results = []
        for i in range(len(logits)):
            result = {
                'scores': top_scores[i],
                'labels': labels[i],
                'boxes': scaled_boxes[i],
            }

            if pred_masks is not None:
                selected_masks_raw = pred_masks[i, query_indices[i]]  # [num_top_queries, H_mask, W_mask]
                masks_probs = selected_masks_raw.sigmoid().unsqueeze(1)   # [num_top_queries, 1, H, W]

                height, width = orig_target_sizes[i].tolist()
                upsampled_masks = F.interpolate(
                    masks_probs,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # [num_top_queries, H, W]

                binary_masks = upsampled_masks > self.mask_threshold
                result['masks'] = binary_masks

            if self.remap_mscoco_category:
                from ...data.coco import mscoco_label2category
                result['labels'] = torch.tensor(
                    [mscoco_label2category[int(lbl)] for lbl in result['labels']],
                    device=labels.device, 
                    dtype=labels.dtype
                )

            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
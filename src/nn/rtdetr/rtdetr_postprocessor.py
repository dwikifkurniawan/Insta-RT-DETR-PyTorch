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
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0) 
        pred_masks = outputs.get('pred_masks', None)       

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # --- 1. Select Top K Predictions for Boxes and Classes ---
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            # scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            scores_flat = scores.flatten(1)
            num_top = min(self.num_top_queries, scores_flat.shape[1])
            top_scores, top_indices = torch.topk(scores_flat, num_top, dim=-1)
            
            labels = mod(top_indices, self.num_classes)
            # These are the indices of the queries that were selected
            query_indices = top_indices // self.num_classes
            
            # Use batch-wise gathering to select the corresponding boxes and masks
            batch_indices = torch.arange(len(logits), device=logits.device).unsqueeze(1)
            selected_boxes = boxes[batch_indices, query_indices]

            selected_masks_raw = None
            if pred_masks is not None:
                # This part is already correct and batched
                batch_indices = torch.arange(len(logits), device=logits.device).unsqueeze(1)
                selected_masks_raw = pred_masks[batch_indices, query_indices]
        else: # This logic is for softmax-based loss, less common for DETR variants now
            scores_softmax, labels = F.softmax(logits, dim=-1)[:, :, :-1].max(-1)
            if scores_softmax.shape[1] > self.num_top_queries:
                top_scores, top_query_indices = torch.topk(scores_softmax, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, 1, top_query_indices)
                
                batch_indices = torch.arange(len(logits), device=logits.device).unsqueeze(1)
                selected_boxes = torch.gather(boxes, 1, top_query_indices.unsqueeze(-1).expand(-1, -1, 4))
                if pred_masks is not None:
                    selected_masks_raw = torch.gather(pred_masks, 1, top_query_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, pred_masks.shape[2], pred_masks.shape[3]))
            else:
                top_scores, selected_boxes, labels = scores_softmax, boxes, labels
                selected_masks_raw = pred_masks
        
        # --- 2. Scale Boxes to Original Image Size ---
        scaled_boxes = torchvision.ops.box_convert(selected_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        img_h = orig_target_sizes[:, 0:1]
        img_w = orig_target_sizes[:, 1:2]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=2)
        scaled_boxes *= scale_fct

        # --- 3. Process and Upsample Masks ---
        processed_masks = None
        if selected_masks_raw is not None:
            # Apply sigmoid in a batched manner
            masks_probs = selected_masks_raw.sigmoid()
            
            # The upsampling part still requires a loop if sizes differ,
            # but we will format the output list directly.
            # The final results loop will handle assigning these masks.
            processed_masks = []
            for i in range(len(logits)):
                # Upsample masks for the i-th image
                upsampled = F.interpolate(
                    masks_probs[i].unsqueeze(0), # Needs a batch dim for interpolate
                    size=orig_target_sizes[i].tolist(),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) # Remove the temp batch dim

                # Binarize and store
                processed_masks.append(upsampled > self.mask_threshold)

        # --- 4. Format Final Results ---
        # `deploy_mode` for ONNX export would need special handling for masks and is omitted for clarity
        if self.deploy_mode:
            return labels, scaled_boxes, top_scores

        results = []
        for i in range(len(logits)):
            result_dict = {
                'scores': top_scores[i],
                'labels': labels[i],
                'boxes': scaled_boxes[i],
            }
            if processed_masks is not None:
                # Assign the pre-processed masks for the i-th image
                result_dict['masks'] = processed_masks[i]
            
            # Remap COCO categories if necessary
            if self.remap_mscoco_category:
                from ...data.coco import mscoco_label2category
                result_dict['labels'] = torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in result_dict['labels'].flatten()],
                    dtype=result_dict['labels'].dtype, 
                    device=result_dict['labels'].device
                ).reshape(result_dict['labels'].shape)

            results.append(result_dict)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
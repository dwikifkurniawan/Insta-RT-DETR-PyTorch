"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.cuda.amp import autocast

from scipy.optimize import linear_sum_assignment
from typing import Dict 

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .rtdetr_criterion import point_sample

# dari MaskDINO
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0, num_points=12544):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        # self.cost_class = weight_dict['cost_class']
        # self.cost_bbox = weight_dict['cost_bbox']
        # self.cost_giou = weight_dict['cost_giou']
        self.cost_class = weight_dict.get('cost_class', 0.0)
        self.cost_bbox = weight_dict.get('cost_bbox', 0.0)
        self.cost_giou = weight_dict.get('cost_giou', 0.0)
        self.cost_dice = weight_dict.get('cost_dice', 0.0)
        self.cost_mask = weight_dict.get('cost_mask', 0.0)

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.num_points = num_points  # ngikutin MaskDINO & Mask2Former

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0 or self.cost_dice != 0 or self.cost_mask != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Iterate through batch size for memory efficiency (from MaskDINO)
        indices = []
        for b in range(bs):

            out_prob_b = outputs["pred_logits"][b]
            tgt_ids_b = targets[b]["labels"]

            if self.use_focal_loss:
                out_prob_b = out_prob_b.sigmoid()
                # Select probabilities for target classes
                prob = out_prob_b[:, tgt_ids_b]
                # Focal loss calculation
                neg_cost_class = (1 - self.alpha) * (prob ** self.gamma) * (-(1 - prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - prob) ** self.gamma) * (-(prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                # Fallback to original softmax-based cost
                out_prob_b = out_prob_b.softmax(-1)
                cost_class = -out_prob_b[:, tgt_ids_b]

            out_bbox = outputs["pred_boxes"][b]  # [num_queries, 4]
            tgt_bbox = targets[b]["boxes"]  # [num_target_boxes, 4]
            
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            
            # Compute mask cost + dice cost
            if "pred_masks" in outputs:
                out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
                # gt masks are already padded when preparing target
                tgt_mask = targets[b]["masks"].to(out_mask)

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=out_mask.dtype)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                # with autocast(enabled=False):
                with torch.amp.autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # If there's no annotations
                    if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                        # Compute the focal loss between masks
                        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                        # Compute the dice loss betwen masks
                        cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    else:
                        cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                        cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            else:
                cost_mask = torch.tensor(0).to(out_bbox)
                cost_dice = torch.tensor(0).to(out_bbox)
        
            # Final cost matrix
            # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_dice * cost_dice + self.cost_mask * cost_mask)
            # C = C.view(bs, num_queries, -1).cpu()

            # sizes = [len(v["boxes"]) for v in targets]
            # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            # handle nan and inf values in cost matrix
            C = torch.nan_to_num(C, nan=100000.0, posinf=100000.0, neginf=100000.0)

            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        final_indices_list = []
        for pred_numpy, tgt_numpy in indices:
            pred_tensor = torch.as_tensor(pred_numpy, dtype=torch.int64)
            tgt_tensor = torch.as_tensor(tgt_numpy, dtype=torch.int64)
            final_indices_list.append((pred_tensor, tgt_tensor))

        return {'indices': final_indices_list}
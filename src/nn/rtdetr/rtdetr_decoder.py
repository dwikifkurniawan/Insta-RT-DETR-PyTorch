"""
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from torch.utils.checkpoint import checkpoint



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.linear1)
    #     linear_init_(self.linear2)
    #     xavier_uniform_(self.linear1.weight)
    #     xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class RTDETRTransformer(nn.Module):
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[256, 512, 1024, 2048],
                 feat_strides=[4, 8, 16, 32],
                 num_levels=4,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                #  box_noise_scale=1.0,
                 box_noise_scale=0.4,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True,
                 mask_head=True):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_det_levels = num_levels - 1 # Levels for detection (P3,P4,P5)
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.mask_head = mask_head

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        # decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points)
        # self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, self.num_det_levels, num_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        # # mask head
        # if self.mask_head:
        #     self.mask_head = nn.ModuleList()
        #     for _ in range(num_layers):
        #         self.mask_head.append(
        #             nn.Sequential(
        #                 nn.Conv2d(hidden_dim + hidden_dim, 256, kernel_size=3, stride=1, padding=1),
                        
        #             )
        #         )
        
        # new mask head
        if self.mask_head:
            self.mask_query_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

        # # init mask head
        if self.mask_head:
            for p in self.mask_query_embed.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            # for m in [self.mask_query_embed, self.mask_pixel_embed]:
            #     for p in m.parameters():
            #         if p.dim() > 1:
            #             nn.init.xavier_uniform_(p)

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        # modified from dn-detr. You can refer to dn-detr
        # https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        mostly copied from MaskDINO. refer to MaskDINO
        https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/transformer_decoder/maskdino_decoder.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
        """
        if self.training:
            scalar, noise_scale = self.num_denoising,self.box_noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//(int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of bbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            # input_label_embed = self.label_enc(m)
            input_label_embed = self.denoising_class_embed(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 4).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label = padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_bbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label, input_query_bbox, attn_mask, mask_dict
    
    def dn_post_process(self, outputs_class, outputs_coord, mask_dict, outputs_mask):
        """
        mostly copied from MaskDINO. refer to MaskDINO
        https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/transformer_decoder/maskdino_decoder.py
        post process of dn after output from the transformer
        put the dn part in the mask_dict
        """
        # assert mask_dict['pad_size'] > 0
        if mask_dict is None:
            return outputs_class, outputs_coord, outputs_mask

        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]

        # ini denoising loss dict
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if output_known_mask is not None:
            out['pred_masks'] = output_known_mask[-1]

        # ini auxiliary output untuk denoising
        # out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask, output_known_coord)
        out['aux_outputs'] = self._set_aux_loss(output_known_class[:-1], output_known_coord[:-1], output_known_mask[:-1] if output_known_mask is not None else None)

        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_coord, outputs_mask

    def _build_input_proj_layer(self, feat_channels):
        # self.input_proj = nn.ModuleList()
        # for in_channels in feat_channels:
        #     self.input_proj.append(
        #         nn.Sequential(OrderedDict([
        #             ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
        #             ('norm', nn.BatchNorm2d(self.hidden_dim,))])
        #         )
        #     )

        # in_channels = feat_channels[-1]

        # for _ in range(self.num_levels - len(feat_channels)):
        #     self.input_proj.append(
        #         nn.Sequential(OrderedDict([
        #             ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
        #             ('norm', nn.BatchNorm2d(self.hidden_dim))])
        #         )
        #     )
        #     in_channels = self.hidden_dim

        # This function is now used to project the 4 output levels from the HybridEncoder
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )


    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        mask_feature = proj_feats[0]
        det_feats = proj_feats[1:]
        # if self.num_levels > len(proj_feats):
        #     len_srcs = len(proj_feats)
        #     for i in range(len_srcs, self.num_levels):
        #         if i == len_srcs:
        #             proj_feats.append(self.input_proj[i](feats[-1]))
        #         else:
        #             proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # Features for query selection (P3, P4, P5 only)
        query_selection_feats = torch.cat([f.flatten(2).permute(0, 2, 1) for f in det_feats], 1)

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for feat in det_feats:
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        # return (feat_flatten, spatial_shapes, level_start_index)
        return feat_flatten, spatial_shapes, level_start_index, mask_feature, query_selection_feats

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides[1:] # for P3, P4, P5
                # for s in [32,32,32]
            ]
            
        anchors = []
        valid_mask = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl+1)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    # def _get_decoder_input(self,
    #                        memory,
    #                        spatial_shapes,
    #                        denoising_class=None,
    #                        denoising_bbox_unact=None):
    #     bs, _, _ = memory.shape
    #     # prepare input for decoder
    #     if self.training or self.eval_spatial_size is None:
    #         anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
    #     else:
    #         anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

    #     # memory = torch.where(valid_mask, memory, 0)
    #     memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

    #     output_memory = self.enc_output(memory)

    #     enc_outputs_class = self.enc_score_head(output_memory)
    #     enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

    #     _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        
    #     reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
    #         index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

    #     enc_topk_bboxes = F.sigmoid(reference_points_unact)
    #     if denoising_bbox_unact is not None:
    #         reference_points_unact = torch.concat(
    #             [denoising_bbox_unact, reference_points_unact], 1)
        
    #     enc_topk_logits = enc_outputs_class.gather(dim=1, \
    #         index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

    #     # extract region features
    #     if self.learnt_init_query:
    #         target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
    #     else:
    #         target = output_memory.gather(dim=1, \
    #             index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
    #         target = target.detach()

    #     if denoising_class is not None:
    #         target = torch.concat([denoising_class, target], 1)

    #     return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def _get_decoder_input(self, memory_for_selection, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        # memory_for_selection is from P3,P4,P5
        bs, _, _ = memory_for_selection.shape
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory_for_selection.device)
        else:
            anchors, valid_mask = self.anchors.to(memory_for_selection.device), self.valid_mask.to(memory_for_selection.device)

        memory = valid_mask.to(memory_for_selection.dtype) * memory_for_selection
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits


    def forward(self, feats, targets=None):
        mask_dict = None
        attn_mask = None
        
        # input projection and embedding
        # (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats) # flatten 
        (memory, det_spatial_shapes, level_start_index, mask_feature, query_selection_feats) = self._get_encoder_input(feats)

        bs = query_selection_feats.shape[0]

        # # prepare denoising training
        # if self.training and self.num_denoising > 0:
        #     denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
        #         get_contrastive_denoising_training_group(targets, \
        #             self.num_classes, 
        #             self.num_queries, 
        #             self.denoising_class_embed, 
        #             num_denoising=self.num_denoising, 
        #             label_noise_ratio=self.label_noise_ratio, 
        #             box_noise_scale=self.box_noise_scale, )
        # else:
        #     denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        #query selectioon
        # target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
        #     self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
        
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(query_selection_feats, det_spatial_shapes)
        
        # if self.training and self.num_denoising > 0:
        #     input_query_label, input_query_bbox, attn_mask, mask_dict = \
        #         self.prepare_for_dn(targets, None, None, bs)
        # if mask_dict is not None and self.training:
        #     target = torch.concat([input_query_label, target], 1)
        #     init_ref_points_unact = torch.concat([input_query_bbox, init_ref_points_unact], 1)

        if self.training:
            dn_results = self.prepare_for_dn(targets, None, None, bs)
            if dn_results is not None:
                dn_label_query, dn_bbox_query, dn_attn_mask, dn_meta = dn_results
                attn_mask = dn_attn_mask
                target = torch.concat([dn_label_query, target], 1)
                init_ref_points_unact = torch.concat([dn_bbox_query, init_ref_points_unact], 1)
                mask_dict = dn_meta

        # decoder
        # out_bboxes, out_logits = self.decoder(
        #     target,
        #     init_ref_points_unact,
        #     memory,
        #     spatial_shapes,
        #     level_start_index,
        #     self.dec_bbox_head,
        #     self.dec_score_head,
        #     self.query_pos_head,
        #     attn_mask=attn_mask)
        out_bboxes, out_logits, out_masks = [], [], []
        inter_queries = target
        ref_points = F.sigmoid(init_ref_points_unact)
        ref_points_detach = ref_points.detach()
        # ref_points_detach = F.sigmoid(init_ref_points_unact)
        for i in range(self.num_layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach)

            # Update queries via self- and cross-attention
            # inter_queries = self.decoder.layers[i](inter_queries, ref_points_input, memory, det_spatial_shapes, level_start_index, attn_mask, None, query_pos_embed)
            # inter_queries = self.decoder.layers[i](
            #     inter_queries, 
            #     ref_points_input, 
            #     memory, 
            #     det_spatial_shapes, 
            #     level_start_index, 
            #     attn_mask, 
            #     None, # memory_mask
            #     query_pos_embed
            # )
            inter_queries = checkpoint(
                self.decoder.layers[i],
                inter_queries, 
                ref_points_input, 
                memory, 
                det_spatial_shapes, 
                level_start_index, 
                attn_mask, 
                None, # memory_mask
                query_pos_embed,
                use_reentrant=False 
            )
            
            # Predict boxes and classes
            inter_bbox_unact = self.dec_bbox_head[i](inter_queries) + inverse_sigmoid(ref_points)
            inter_bbox = F.sigmoid(inter_bbox_unact)
            inter_logit = self.dec_score_head[i](inter_queries)
            
            if self.training:
                out_logits.append(inter_logit)
                out_bboxes.append(inter_bbox)

            ref_points = inter_bbox
            ref_points_detach = inter_bbox.detach()

            # Predict masks using the efficient dot-product method
            if self.mask_head:
                mask_query_embedding = self.mask_query_embed(inter_queries)
                mask_logits = torch.einsum('bnc,bchw->bnhw', mask_query_embedding, mask_feature)
                if self.training: 
                    out_masks.append(mask_logits)
        
        # For inference, take only the output of the last layer
        if not self.training:
            out_logits.append(inter_logit)
            out_bboxes.append(inter_bbox)
            if self.mask_head: out_masks.append(mask_logits)

        # Stack predictions from all layers
        out_bboxes = torch.stack(out_bboxes)
        out_logits = torch.stack(out_logits)
        if self.mask_head and out_masks: 
            out_masks = torch.stack(out_masks)
            
        # if self.training and dn_meta is not None:
        #     dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
        #     dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
        #     if self.mask_head and len(out_masks) > 0:
        #         dn_out_masks, out_masks = torch.split(out_masks, dn_meta['dn_num_split'], dim=2)

        out_logits, out_bboxes, out_masks = self.dn_post_process(out_logits, out_bboxes, mask_dict, out_masks)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if self.mask_head and len(out_masks) > 0: 
            out['pred_masks'] = out_masks[-1]

        # Prepare auxiliary outputs for loss calculation
        if self.training and self.aux_loss:
            aux_masks = out_masks[:-1] if self.mask_head and len(out_masks) > 1 else None
            aux_outputs = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1], aux_masks)
            aux_outputs.extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes], None))
            out['aux_outputs'] = aux_outputs

            if mask_dict is not None:
                out['dn_aux_outputs'] = mask_dict['output_known_lbs_bboxes']
                out['dn_meta'] = mask_dict
            
            # if dn_meta is not None:
            #     dn_aux_masks = dn_out_masks if self.mask_head and len(out_masks) > 0 else None
            #     out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes, dn_aux_masks)
            #     out['dn_meta'] = dn_meta

        # if self.training and self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        #     out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
            
        #     if self.training and dn_meta is not None:
        #         out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
        #         out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_mask is not None:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                    for a, b, c in zip(outputs_class, outputs_coord, outputs_mask)]
        else:
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class, outputs_coord)]
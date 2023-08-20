import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.detection.models import backbones
from simpleAICV.detection.models.backbones.detr_resnet import DINOPositionEmbeddingBlock
from simpleAICV.detection.models.deformable_transformer import DeformableTransformer, DINODETRClsHead, DINODETRRegHead


class DINODETR(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path,
                 hidden_inplanes=256,
                 dropout_prob=0.0,
                 head_nums=8,
                 query_nums=900,
                 feedforward_planes=2048,
                 encoder_layer_nums=6,
                 decoder_layer_nums=6,
                 activation='relu',
                 feature_level_nums=5,
                 encoder_point_nums=4,
                 decoder_point_nums=4,
                 module_seq=['sa', 'ca', 'ffn'],
                 num_classes=80,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=80):
        super(DINODETR, self).__init__()
        self.hidden_inplanes = hidden_inplanes
        self.query_nums = query_nums
        self.num_classes = num_classes

        assert num_classes == dn_labelbook_size
        self.label_encoder = nn.Embedding(dn_labelbook_size + 1,
                                          hidden_inplanes)

        # for dn training
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
            })
        self.position_embedding = DINOPositionEmbeddingBlock(
            inplanes=hidden_inplanes // 2,
            temperature_h=10000,
            temperature_w=10000,
            eps=1e-6)

        input_proj_layers = []
        for idx in range(len(self.backbone.out_channels)):
            in_channels = self.backbone.out_channels[idx]
            input_proj_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_inplanes, kernel_size=1),
                    nn.GroupNorm(32, hidden_inplanes),
                ))
        input_proj_layers.append(
            nn.Sequential(
                nn.Conv2d(self.backbone.out_channels[-1],
                          hidden_inplanes,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.GroupNorm(32, hidden_inplanes),
            ))
        self.input_proj_layers = nn.ModuleList(input_proj_layers)

        for per_proj_layer in self.input_proj_layers:
            nn.init.xavier_uniform_(per_proj_layer[0].weight, gain=1)
            nn.init.constant_(per_proj_layer[0].bias, 0)

        self.transformer = DeformableTransformer(
            inplanes=hidden_inplanes,
            dropout_prob=dropout_prob,
            head_nums=head_nums,
            query_nums=query_nums,
            feedforward_planes=feedforward_planes,
            encoder_layer_nums=encoder_layer_nums,
            decoder_layer_nums=decoder_layer_nums,
            activation=activation,
            feature_level_nums=feature_level_nums,
            encoder_point_nums=encoder_point_nums,
            decoder_point_nums=decoder_point_nums,
            module_seq=module_seq,
            num_classes=num_classes)

        self.bbox_embed = nn.ModuleList([
            DINODETRRegHead(hidden_inplanes=hidden_inplanes, num_layers=3)
            for _ in range(decoder_layer_nums)
        ])
        self.class_embed = nn.ModuleList([
            DINODETRClsHead(hidden_inplanes, num_classes)
            for _ in range(decoder_layer_nums)
        ])
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

    def inverse_sigmoid(self, x, eps=1e-4):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)

        return torch.log(x1 / x2)

    def prepare_for_dn(self, targets, dn_number, dn_label_noise_ratio,
                       dn_box_noise_scale, query_nums, num_classes, hidden_dim,
                       label_enc):
        """
            A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
            forward function and use learnable tgt embedding, so we change this function a little bit.
            :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
            :param training: if it is training or inference
            :param query_nums: number of queires
            :param num_classes: number of classes
            :param hidden_dim: transformer hidden dim
            :param label_enc: encode labels in dn
            :return:
            """
        device = targets.device

        # positive and negative dn queries
        dn_number = dn_number * 2

        filter_targets = []
        for per_image_targets in targets:
            per_image_targets = per_image_targets[per_image_targets[:, 4] >= 0]
            filter_targets.append(per_image_targets)

        known = [(torch.ones_like(per_image_targets[:, 4]).to(device))
                 for per_image_targets in filter_targets]

        batch_size = len(known)
        known_num = [sum(k) for k in known]

        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)

        labels = torch.cat([
            per_image_targets[:, 4] for per_image_targets in filter_targets
        ]).to(device)
        boxes = torch.cat([
            per_image_targets[:, 0:4] for per_image_targets in filter_targets
        ]).to(device)
        batch_idx = torch.cat([
            torch.full_like(per_image_targets[:, 4].long(), idx)
            for idx, per_image_targets in enumerate(filter_targets)
        ]).to(device)

        known_indice = torch.nonzero(unmask_label + unmask_bbox).to(device)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if dn_label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(
                p < (dn_label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes).type_as(known_labels_expaned)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(
            len(boxes))).long().to(device).unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) *
                         2).long().to(device).unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        if dn_box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs).to(device)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(
                known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs).to(device)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(
                rand_part, diff).to(device) * dn_box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] +
                                        known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to(device)
        input_label_embed = label_enc(m)
        input_bbox_embed = self.inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).to(device)
        padding_bbox = torch.zeros(pad_size, 4).to(device)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to(device)
        if len(known_num):
            map_known_indice = torch.cat([
                torch.tensor(range(int(num))) for num in known_num
            ]).to(device)  # [1,2, 1,2,3]
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i for i in range(2 * dn_number)
            ]).long().to(device)
        if len(known_bid):
            input_query_label[(known_bid.long(),
                               map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(),
                              map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + query_nums
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 *
                          (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 *
                          (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_meta):
        """
            post process of dn after output from the transformer
            put the dn part in the dn_meta
        """
        if dn_meta and dn_meta['pad_size'] > 0:
            output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
            output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
            outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
            outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
            out = {
                'pred_logits': output_known_class[-1],
                'pred_boxes': output_known_coord[-1]
            }

            out['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(output_known_class[:-1], output_known_coord[:-1])
                                  ]

            dn_meta['output_known_lbs_bboxes'] = out

        return outputs_class, outputs_coord

    def forward(self, inputs, masks, targets=None):
        assert masks is not None

        features = self.backbone(inputs)

        resized_masks = []
        resized_positions = []
        for per_feature in features:
            per_mask = F.interpolate(
                masks.float().unsqueeze(1),
                size=[per_feature.shape[2],
                      per_feature.shape[3]]).to(torch.bool).squeeze(1)
            assert per_mask is not None
            resized_masks.append(per_mask)

            per_position = self.position_embedding(per_mask)
            resized_positions.append(per_position)

        assert len(features) == len(resized_masks) == len(resized_positions)

        total_features = []
        total_masks = []
        total_positions = []
        for idx, (per_feature, per_mask, per_position) in enumerate(
                zip(features, resized_masks, resized_positions)):
            total_features.append(self.input_proj_layers[idx](per_feature))
            total_masks.append(per_mask)
            total_positions.append(per_position)

        assert len(total_features) == len(total_masks) == len(total_positions)

        level_5_feature = self.input_proj_layers[-1](features[-1])
        level_5_mask = F.interpolate(
            masks.float().unsqueeze(1),
            size=[level_5_feature.shape[2],
                  level_5_feature.shape[3]]).to(torch.bool).squeeze(1)
        level_5_position = self.position_embedding(level_5_mask)
        total_features.append(level_5_feature)
        total_masks.append(level_5_mask)
        total_positions.append(level_5_position)

        assert len(total_features) == len(total_masks) == len(total_positions)

        del features, resized_masks, resized_positions

        if targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_dn(
                targets,
                self.dn_number,
                self.dn_label_noise_ratio,
                self.dn_box_noise_scale,
                query_nums=self.query_nums,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_inplanes,
                label_enc=self.label_encoder)
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            total_features, total_masks, input_query_bbox, total_positions,
            input_query_label, attn_mask)

        # In case num object=0
        hs[0] += self.label_encoder.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + self.inverse_sigmoid(
                layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack([
            layer_cls_embed(layer_hs)
            for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
        ])
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = self.dn_post_process(
                outputs_class, outputs_coord_list, dn_meta)
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord_list[-1]
        }
        out['aux_outputs'] = [{
            'pred_logits': a,
            'pred_boxes': b
        } for a, b in zip(outputs_class[:-1], outputs_coord_list[:-1])]

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {
                'pred_logits': interm_class,
                'pred_boxes': interm_coord
            }

        out['dn_meta'] = dn_meta

        return out


def _dinodetr(backbone_type, backbone_pretrained_path, **kwargs):
    model = DINODETR(backbone_type,
                     backbone_pretrained_path=backbone_pretrained_path,
                     **kwargs)

    return model


def resnet18_dinodetr(backbone_pretrained_path='', **kwargs):
    return _dinodetr('detr_resnet18backbone',
                     backbone_pretrained_path=backbone_pretrained_path,
                     **kwargs)


def resnet34_dinodetr(backbone_pretrained_path='', **kwargs):
    return _dinodetr('detr_resnet34backbone',
                     backbone_pretrained_path=backbone_pretrained_path,
                     **kwargs)


def resnet50_dinodetr(backbone_pretrained_path='', **kwargs):
    return _dinodetr('detr_resnet50backbone',
                     backbone_pretrained_path=backbone_pretrained_path,
                     **kwargs)


def resnet101_dinodetr(backbone_pretrained_path='', **kwargs):
    return _dinodetr('detr_resnet101backbone',
                     backbone_pretrained_path=backbone_pretrained_path,
                     **kwargs)


def resnet152_dinodetr(backbone_pretrained_path='', **kwargs):
    return _dinodetr('detr_resnet152backbone',
                     backbone_pretrained_path=backbone_pretrained_path,
                     **kwargs)


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from tools.path import COCO2017_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.detection.datasets.cocodataset import CocoDetection
    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DetectionResize, DetectionCollater, DETRDetectionCollater

    cocodataset = CocoDetection(COCO2017_path,
                                set_name='train2017',
                                transform=transforms.Compose([
                                    RandomHorizontalFlip(prob=0.5),
                                    RandomCrop(prob=0.5),
                                    RandomTranslate(prob=0.5),
                                    DetectionResize(
                                        resize=800,
                                        stride=32,
                                        resize_type='yolo_style',
                                        multi_scale=False,
                                        multi_scale_range=[0.8, 1.0]),
                                    Normalize(),
                                ]))

    from torch.utils.data import DataLoader
    collater = DETRDetectionCollater(resize=800,
                                     resize_type='yolo_style',
                                     max_annots_num=100)
    train_loader = DataLoader(cocodataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    net = resnet50_dinodetr()

    for data in tqdm(train_loader):
        images, annots, masks, scales, sizes = data['image'], data[
            'annots'], data['mask'], data['scale'], data['size']
        print('0000', images.shape, annots.shape, masks.shape, scales.shape,
              sizes.shape)
        print('0000', images.dtype, annots.dtype, masks.dtype, scales.dtype,
              sizes.dtype)

        net = net.cuda()
        images = images.cuda()
        masks = masks.cuda()

        from thop import profile
        from thop import clever_format
        macs, params = profile(net, inputs=(images, masks), verbose=False)
        macs, params = clever_format([macs, params], '%.3f')
        print(f'1111, macs: {macs}, params: {params}')

        # outs = net(torch.autograd.Variable(images),
        #            torch.autograd.Variable(masks))
        # print('2222', outs.keys())

        break

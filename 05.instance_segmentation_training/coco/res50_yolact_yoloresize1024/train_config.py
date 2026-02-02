import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path

from SimpleAICV.instance_segmentation import models
from SimpleAICV.instance_segmentation import losses
from SimpleAICV.instance_segmentation import decode
from SimpleAICV.instance_segmentation.datasets.cocodataset import CocoInstanceSegmentation
from SimpleAICV.instance_segmentation.common import InstanceSegmentationResize, RandomHorizontalFlip, Normalize, YOLACTInstanceSegmentationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_yolact'
    num_classes = 80
    input_image_size = [1024, 1024]

    # load backbone pretrained model or not
    backbone_pretrained_path = '/root/autodl-tmp/pretrained_models/resnet_convert_from_pytorch_official_weights/resnet50-11ad3fa6-acc1-80.858_pytorch_official_weight_convert.pth'
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes + 1,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    train_criterion = losses.__dict__['YOLACTLoss'](
        **{
            'resize': input_image_size[0],
            'resize_type': 'yolo_style',
            'scales': [24, 48, 96, 192, 384],
            'ratios': [1, 1 / 2, 2],
            'strides': [8, 16, 32, 64, 128],
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'mask_loss_weight': 1.,
            'semantic_seg_loss_weight': 1.,
        })
    test_criterion = losses.__dict__['YOLACTLoss'](
        **{
            'resize': input_image_size[0],
            'resize_type': 'yolo_style',
            'scales': [24, 48, 96, 192, 384],
            'ratios': [1, 1 / 2, 2],
            'strides': [8, 16, 32, 64, 128],
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'mask_loss_weight': 1.,
            'semantic_seg_loss_weight': 1.,
        })

    decoder = decode.__dict__['YOLACTDecoder'](**{
        'resize': input_image_size[0],
        'resize_type': 'yolo_style',
        'scales': [24, 48, 96, 192, 384],
        'ratios': [1, 1 / 2, 2],
        'strides': [8, 16, 32, 64, 128],
        'max_feature_upsample_scale': 8,
        'topn': 200,
        'max_object_num': 100,
        'min_score_threshold': 0.05,
        'nms_threshold': 0.5,
    }).cuda()

    train_dataset = CocoInstanceSegmentation(
        COCO2017_path,
        set_name='train2017',
        filter_no_object_image=True,
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=input_image_size[0],
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=True,
                                       multi_scale_range=[0.8, 1.0]),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    test_dataset = CocoInstanceSegmentation(
        COCO2017_path,
        set_name='val2017',
        filter_no_object_image=True,
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=input_image_size[0],
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=False,
                                       multi_scale_range=[0.8, 1.0]),
            Normalize(),
        ]))

    train_collater = YOLACTInstanceSegmentationCollater(
        resize=input_image_size[0], resize_type='yolo_style')
    test_collater = YOLACTInstanceSegmentationCollater(
        resize=input_image_size[0], resize_type='yolo_style')

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 32
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-3,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 1,
            'gamma': 0.1,
            'milestones': [24, 36],
        },
    )

    epochs = 39
    print_interval = 100

    eval_type = 'COCO'
    save_model_metric = 'IoU=0.50:0.95,area=all,maxDets=100,mAP'

    sync_bn = False
    use_amp = True
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }

    use_ema_model = False
    ema_model_decay = 0.9999

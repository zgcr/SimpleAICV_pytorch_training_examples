import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import VOCdataset_path

from simpleAICV.detection import models
from simpleAICV.detection import losses
from simpleAICV.detection import decode
from simpleAICV.detection.datasets.vocdataset import VocDetection
from simpleAICV.detection.common import DetectionResize, RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_fcos'
    num_classes = 20
    input_image_size = [640, 640]

    # load backbone pretrained model or not
    backbone_pretrained_path = '/root/code/SimpleAICV_pytorch_training_examples/pretrained_models/resnet_convert_from_pytorch_official_weights/resnet50-11ad3fa6-acc1-80.858_pytorch_official_weight_convert.pth'
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = losses.__dict__['FCOSLoss'](
        **{
            'strides': [8, 16, 32, 64, 128],
            'mi': [[-1, 64], [64, 128], [128, 256], [256, 512],
                   [512, 100000000]],
            'alpha': 0.25,
            'gamma': 2.,
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'center_ness_loss_weight': 1.,
            'box_loss_iou_type': 'GIoU',
            'center_sample_radius': 1.5,
            'use_center_sample': True,
        })
    test_criterion = losses.__dict__['FCOSLoss'](
        **{
            'strides': [8, 16, 32, 64, 128],
            'mi': [[-1, 64], [64, 128], [128, 256], [256, 512],
                   [512, 100000000]],
            'alpha': 0.25,
            'gamma': 2.,
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'center_ness_loss_weight': 1.,
            'box_loss_iou_type': 'GIoU',
            'center_sample_radius': 1.5,
            'use_center_sample': True,
        })

    decoder = decode.__dict__['FCOSDecoder'](**{
        'strides': [8, 16, 32, 64, 128],
        'max_object_num': 100,
        'min_score_threshold': 0.05,
        'topn': 1000,
        'nms_type': 'python_nms',
        'nms_threshold': 0.6,
    })

    train_dataset = VocDetection(root_dir=VOCdataset_path,
                                 image_sets=[('2007', 'trainval'),
                                             ('2012', 'trainval')],
                                 transform=transforms.Compose([
                                     RandomHorizontalFlip(prob=0.5),
                                     RandomCrop(prob=0.5),
                                     RandomTranslate(prob=0.5),
                                     DetectionResize(
                                         resize=input_image_size[0],
                                         stride=32,
                                         resize_type='yolo_style',
                                         multi_scale=True,
                                         multi_scale_range=[0.8, 1.0]),
                                     Normalize(),
                                 ]),
                                 keep_difficult=False)

    test_dataset = VocDetection(root_dir=VOCdataset_path,
                                image_sets=[('2007', 'test')],
                                transform=transforms.Compose([
                                    DetectionResize(
                                        resize=input_image_size[0],
                                        stride=32,
                                        resize_type='yolo_style',
                                        multi_scale=False,
                                        multi_scale_range=[0.8, 1.0]),
                                    Normalize(),
                                ]),
                                keep_difficult=False)

    train_collater = DetectionCollater(resize=input_image_size[0],
                                       resize_type='yolo_style',
                                       max_annots_num=100)
    test_collater = DetectionCollater(resize=input_image_size[0],
                                      resize_type='yolo_style',
                                      max_annots_num=100)

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 16
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
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [8, 12],
        },
    )

    epochs = 13
    print_interval = 100

    eval_type = 'VOC'
    eval_epoch = [1, 3, 5, 8, 10, 12, 13]
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]
    save_model_metric = 'IoU=0.50,area=all,maxDets=100,mAP'

    sync_bn = False
    use_amp = False
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }

    use_ema_model = False
    ema_model_decay = 0.9999

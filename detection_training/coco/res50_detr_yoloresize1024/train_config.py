import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path

from simpleAICV.detection import models
from simpleAICV.detection import losses
from simpleAICV.detection import decode
from simpleAICV.detection.datasets.cocodataset import CocoDetection
from simpleAICV.detection.common import DetectionResize, RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DETRDetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_detr'
    num_classes = 80
    input_image_size = [1024, 1024]

    # load backbone pretrained model or not
    backbone_pretrained_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/resnet_train_from_scratch_on_imagenet1k/resnet50-acc76.300.pth'
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = losses.__dict__['DETRLoss'](**{
        'cls_match_cost': 1.0,
        'box_match_cost': 5.0,
        'giou_match_cost': 2.0,
        'cls_loss_weight': 1.0,
        'box_l1_loss_weight': 5.0,
        'iou_loss_weight': 2.0,
        'no_object_cls_weight': 0.1,
        'num_classes': num_classes,
    })
    test_criterion = losses.__dict__['DETRLoss'](**{
        'cls_match_cost': 1.0,
        'box_match_cost': 5.0,
        'giou_match_cost': 2.0,
        'cls_loss_weight': 1.0,
        'box_l1_loss_weight': 5.0,
        'iou_loss_weight': 2.0,
        'no_object_cls_weight': 0.1,
        'num_classes': num_classes,
    })
    decoder = decode.__dict__['DETRDecoder'](**{
        'num_classes': num_classes,
        'max_object_num': 100,
        'min_score_threshold': 0.05,
        'topn': 100,
        'nms_type': None,
        'nms_threshold': 0.5,
    })

    train_dataset = CocoDetection(COCO2017_path,
                                  set_name='train2017',
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
                                  ]))

    test_dataset = CocoDetection(COCO2017_path,
                                 set_name='val2017',
                                 transform=transforms.Compose([
                                     DetectionResize(
                                         resize=input_image_size[0],
                                         stride=32,
                                         resize_type='yolo_style',
                                         multi_scale=False,
                                         multi_scale_range=[0.8, 1.0]),
                                     Normalize(),
                                 ]))

    train_collater = DETRDetectionCollater(resize=input_image_size[0],
                                           resize_type='yolo_style',
                                           max_annots_num=100)
    test_collater = DETRDetectionCollater(resize=input_image_size[0],
                                          resize_type='yolo_style',
                                          max_annots_num=100)

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 64
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': True,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-4,
            'no_weight_decay_layer_name_list': [],
            'sub_layer_lr': {
                'backbone': 1e-5,
            },
        },
    )

    clip_max_norm = 0.1

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [400],
        },
    )

    epochs = 500
    print_interval = 100

    # 'COCO' or 'VOC'
    eval_type = 'COCO'
    eval_epoch = [1]
    for i in range(epochs):
        if i % 10 == 0:
            eval_epoch.append(i)
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]
    save_model_metric = 'IoU=0.50:0.95,area=all,maxDets=100,mAP'

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
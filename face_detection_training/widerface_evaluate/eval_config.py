import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from simpleAICV.face_detection import models
from simpleAICV.face_detection import decode
from simpleAICV.face_detection.common import load_state_dict


class config:
    network = 'resnet50_retinaface'
    num_classes = 1
    input_image_size = [960, 960]

    model = models.__dict__[network](**{
        'backbone_pretrained_path': '',
        'anchor_num': 3,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    decoder = decode.__dict__['RetinaFaceDecoder'](
        **{
            'anchor_sizes': [[8, 16, 32], [32, 64, 128], [128, 256, 512]],
            'strides': [8, 16, 32],
            'max_object_num': 200,
            'min_score_threshold': 0.3,
            'topn': 1000,
            'nms_type': 'python_nms',
            'nms_threshold': 0.3,
        })

    seed = 0

    eval_image_dir = '/root/autodl-tmp/face_detection_dataset/wider_face_val_images/images'
    save_image_dir = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/face_detection_training/widerface_evaluate/val_images_result'
    save_image_result = False

    gt_mat_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/face_detection_training/widerface_evaluate/widerface_ground_truth/wider_face_val.mat'
    hard_mat_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/face_detection_training/widerface_evaluate/widerface_ground_truth/wider_hard_val.mat'
    medium_mat_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/face_detection_training/widerface_evaluate/widerface_ground_truth/wider_medium_val.mat'
    easy_mat_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/face_detection_training/widerface_evaluate/widerface_ground_truth/wider_easy_val.mat'

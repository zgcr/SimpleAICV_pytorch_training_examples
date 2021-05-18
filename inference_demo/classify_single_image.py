import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from simpleAICV.classification import backbones
from tools.utils import compute_flops_and_params


def parse_args():
    parser = argparse.ArgumentParser(description='detect image')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--model', type=str, help='name of model')
    parser.add_argument('--trained_num_classes',
                        type=int,
                        default=1000,
                        help='model class num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=224,
                        help='input image size')
    parser.add_argument('--trained_model_path',
                        type=str,
                        default='',
                        help='trained model path')
    parser.add_argument('--test_image_path', type=str, help='test image path')
    parser.add_argument("--save_image_path",
                        type=str,
                        help="save detected image path")
    parser.add_argument('--show_image',
                        default=False,
                        action='store_true',
                        help='show_image or not')
    parser.add_argument('--use_gpu',
                        default=False,
                        action='store_true',
                        help='use gpu to test or not')
    args = parser.parse_args()

    return args


def inference():
    args = parse_args()
    print(f'args: {args}')

    assert args.model in backbones.__dict__.keys(), 'Unsupported model!'

    if args.use_gpu:
        # only use one Graphics card to inference
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        assert torch.cuda.is_available(), 'need gpu to train network!'
        torch.cuda.empty_cache()

    if args.seed:
        seed = args.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.use_gpu:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # for cudnn
            cudnn.enabled = True
            cudnn.deterministic = True
            cudnn.benchmark = False

    model = backbones.__dict__[args.model](
        **{
            'num_classes': args.trained_num_classes,
        })

    if args.use_gpu:
        model = model.cuda()

    if args.trained_model_path:
        saved_model = torch.load(args.trained_model_path,
                                 map_location=torch.device('cpu'))
        model.load_state_dict(saved_model)

    model.eval()

    flops, params = compute_flops_and_params(args, model)
    print(f'model: {args.model}, flops: {flops}, params: {params}')

    origin_img = Image.open(args.test_image_path)
    img = origin_img
    transform = transforms.Compose([
        transforms.Resize(int(args.input_image_size * (256 / 224))),
        transforms.CenterCrop(args.input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = torch.tensor(img)

    if args.use_gpu:
        img = img.cuda()

    origin_img = cv2.cvtColor(np.asarray(origin_img), cv2.COLOR_RGB2BGR)

    output = model(img.unsqueeze(0))
    output = F.softmax(output, dim=1)
    pred_score, pred_class = output.max(dim=1)
    pred_score, pred_class = pred_score.item(), pred_class.item()
    color = [random.randint(0, 255) for _ in range(3)]
    print(f'score: {pred_score:.3f}, class: {pred_class}, color: {color}')

    text = f'{pred_class}:{pred_score:.3f}'
    cv2.putText(origin_img,
                text, (30, 30),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                color=color,
                thickness=1)

    if args.save_image_path:
        cv2.imwrite(
            os.path.join(args.save_image_path, 'classification_result.jpg'),
            origin_img)

    if args.show_image:
        cv2.namedWindow("classification_result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('classification_result', origin_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    inference()

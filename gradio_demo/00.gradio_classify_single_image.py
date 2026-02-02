import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import gradio as gr

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from SimpleAICV.classification import backbones
from SimpleAICV.classification.common import load_state_dict
from tools.utils import set_seed


class config:
    network = 'resnet50'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/resnet_finetune_on_imagenet1k_from_imagenet21k_pretrain/resnet50-acc80.110.pth'
    load_state_dict(trained_model_path, model)

    seed = 0


@torch.no_grad()
def predict(image):
    set_seed(config.seed)

    transform = transforms.Compose([
        transforms.Resize(int(config.input_image_size * config.scale)),
        transforms.CenterCrop(config.input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = torch.tensor(image).unsqueeze(0)

    model = config.model

    model.eval()

    with torch.no_grad():
        output = F.softmax(model(image), dim=1)
        output = output.squeeze(0)

    return {f'类别{i}': float(output[i]) for i in range(config.num_classes)}


title = '图像分类demo'
description = '选择一张图片进行图像分类吧！'
inputs = gr.Image(type='pil')
outputs = gr.Label(num_top_classes=5)
gradio_demo = gr.Interface(fn=predict,
                           title=title,
                           description=description,
                           inputs=inputs,
                           outputs=outputs,
                           examples=[
                               'test_coco_images/000000001551.jpg',
                               'test_coco_images/000000010869.jpg',
                               'test_coco_images/000000011379.jpg',
                               'test_coco_images/000000015108.jpg',
                               'test_coco_images/000000016656.jpg',
                           ])
# local website: http://127.0.0.1:6006/
gradio_demo.launch(share=True,
                   server_name='0.0.0.0',
                   server_port=6006,
                   show_error=True)

import os
import sys
import warnings

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import gradio as gr
import random
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from simpleAICV.classification import backbones
from simpleAICV.classification.common import load_state_dict

seed = 0
model_name = 'resnet50'
model_num_classes = 1000
trained_model_path = '/root/autodl-tmp/pretrained_models/resnet_finetune_on_imagenet1k_from_imagenet21k_pretrain/resnet50-acc80.258.pth'
input_image_size = 224

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

assert model_name in backbones.__dict__.keys(), 'Unsupported model!'
model = backbones.__dict__[model_name](**{
    'num_classes': model_num_classes,
})
if trained_model_path:
    load_state_dict(trained_model_path, model)
else:
    print('No pretrained model load!')
model.eval()


@torch.no_grad
def predict(image):
    transform = transforms.Compose([
        transforms.Resize(int(input_image_size * (256 / 224))),
        transforms.CenterCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = torch.tensor(image).unsqueeze(0)

    with torch.no_grad():
        output = F.softmax(model(image), dim=1)
        output = output.squeeze(0)

    return {f'类别{i}': float(output[i]) for i in range(model_num_classes)}


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

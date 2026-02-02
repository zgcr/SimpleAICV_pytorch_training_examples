import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleAICV.interactive_segmentation.models.segment_anything import sam
from SimpleAICV.interactive_segmentation.common import load_state_dict
from tools.utils import set_seed


class config:
    network = 'sam_h'
    input_image_size = 1024

    model = sam.__dict__[network](**{
        'image_size': input_image_size,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/sam_pytorch_official_weights/sam_vit_h_4b8939.pth'
    load_state_dict(trained_model_path, model)

    seed = 0


def preprocess_image(image, resize):
    # PIL image(RGB) to opencv image(RGB)
    image = np.asarray(image).astype(np.float32)

    origin_image = image.copy()
    h, w, _ = origin_image.shape

    origin_size = [h, w]

    factor = resize / max(h, w)

    resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
    image = cv2.resize(image, (resize_w, resize_h))

    # normalize
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    image = (image - mean) / std

    padded_img = np.zeros(
        (max(resize_h, resize_w), max(resize_h, resize_w), 3),
        dtype=np.float32)
    padded_img[:resize_h, :resize_w, :] = image

    scale = factor
    scaled_size = [resize_h, resize_w]

    return origin_image, padded_img, scale, scaled_size, origin_size


@torch.no_grad()
def predict(inputs, mask_out_idx):
    set_seed(config.seed)

    # 处理 ImageEditor 的输出格式
    if inputs is None:
        return None, None

    # ImageEditor 返回字典格式 {'background': ..., 'layers': [...], 'composite': ...}
    image = inputs['background']
    layers = inputs['layers']

    if image is None:
        return None, None

    if layers is None or len(layers) == 0:
        return None, None

    # 转换为 numpy 数组
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 如果是 RGBA，转换为 RGB
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    origin_image, resized_img, scale, scaled_size, origin_size = preprocess_image(
        image, config.input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    # 用户的笔刷绘制会在 layers[0] 中
    mask = None
    # 从第一个 layer 提取 mask
    layer = layers[0]
    if isinstance(layer, Image.Image):
        layer = np.array(layer)

    # layer 通常是 RGBA 格式，alpha 通道表示绘制的区域
    if layer.shape[-1] == 4:
        # 使用 alpha 通道作为 mask
        mask = layer[:, :, 3]
    else:
        # 如果没有 alpha 通道，检查非零像素
        mask = np.any(layer > 0, axis=2).astype(np.uint8) * 255

    if mask is None or np.sum(mask) == 0:
        return None, None

    # 获取最小外接矩形坐标
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    x1, y1, w, h = cv2.boundingRect(mask)
    x2 = x1 + w
    y2 = y1 + h

    input_box = np.array([x1, y1, x2, y2]) * scale
    input_prompt_box = torch.tensor(np.expand_dims(input_box,
                                                   axis=0)).float().cuda()

    batch_prompts = {
        'prompt_point': None,
        'prompt_box': input_prompt_box,
        'prompt_mask': None
    }

    mask_out_idx = [mask_out_idx]

    model = config.model

    model.eval()

    with torch.no_grad():
        mask_preds, iou_preds = model(resized_img,
                                      batch_prompts,
                                      mask_out_idxs=mask_out_idx)
        mask_preds, iou_preds = mask_preds[0][0], iou_preds[0][0]
        binary_mask_preds = mask_preds > 0.

    masks = binary_mask_preds.numpy().astype(np.float32)
    masks = masks[:scaled_size[0], :scaled_size[1]]

    iou_preds = iou_preds.numpy()

    masks = cv2.resize(masks, (origin_size[1], origin_size[0]),
                       interpolation=cv2.INTER_NEAREST)

    binary_mask = (masks.copy() * 255.).astype(np.uint8)

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    origin_image = origin_image.astype(np.uint8)

    masks_class_color = list(np.random.choice(range(256), size=3))

    per_image_mask = np.zeros(
        (origin_image.shape[0], origin_image.shape[1], 3))

    per_image_contours = []
    per_mask = masks

    per_mask_color = np.array(
        (masks_class_color[0], masks_class_color[1], masks_class_color[2]))

    per_object_mask = np.nonzero(per_mask == 1.)
    per_image_mask[per_object_mask[0], per_object_mask[1]] = per_mask_color

    # get contours
    new_per_image_mask = np.zeros(
        (origin_image.shape[0], origin_image.shape[1]))
    new_per_image_mask[per_object_mask[0], per_object_mask[1]] = 255
    contours, _ = cv2.findContours(new_per_image_mask.astype(np.uint8),
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    per_image_contours.append(contours)

    per_image_mask = per_image_mask.astype(np.uint8)
    per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)

    all_object_mask = np.nonzero(per_image_mask != 0)
    per_image_mask[all_object_mask[0], all_object_mask[1]] = cv2.addWeighted(
        origin_image[all_object_mask[0], all_object_mask[1]], 0.5,
        per_image_mask[all_object_mask[0], all_object_mask[1]], 1, 0)
    no_class_mask = np.nonzero(per_image_mask == 0)
    per_image_mask[no_class_mask[0],
                   no_class_mask[1]] = origin_image[no_class_mask[0],
                                                    no_class_mask[1]]
    for contours in per_image_contours:
        cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 1)

    per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_BGR2RGB)
    per_image_mask = Image.fromarray(np.uint8(per_image_mask))

    return per_image_mask, binary_mask


with gr.Blocks() as demo:
    with gr.Tab(label='Segment Anything!'):
        with gr.Row():
            with gr.Column():
                inputs = gr.ImageEditor(
                    label=
                    'Circle the target (Use brush to draw, eraser to erase)',
                    type='numpy',
                    brush=gr.Brush(default_size=30,
                                   colors=["#FF0000"]),  # 红色笔刷
                    eraser=gr.Eraser(default_size=30)  # 橡皮擦工具
                )
                with gr.Row():
                    gr.Markdown('Choose sam model mask out idx.')
                    mask_out_idx = gr.Slider(minimum=0,
                                             maximum=3,
                                             value=0,
                                             step=1,
                                             label='mask out idx')

                # 按钮组
                with gr.Row():
                    run_button = gr.Button('RUN!', variant='primary')
                    clear_button = gr.Button('Clear Drawing',
                                             variant='secondary')

            # show image with mask
            with gr.Tab(label='Image with Mask'):
                output_image_with_mask = gr.Image(type='pil')
            # only show mask
            with gr.Tab(label='Mask'):
                output_mask = gr.Image(type='pil')

    def clear_editor():
        return None

    # 运行按钮点击事件
    run_button.click(predict,
                     inputs=[inputs, mask_out_idx],
                     outputs=[output_image_with_mask, output_mask])

    # 清除按钮点击事件
    clear_button.click(clear_editor, outputs=inputs)

# local website: http://127.0.0.1:6006/
demo.queue().launch(share=True,
                    server_name='0.0.0.0',
                    server_port=6006,
                    show_error=True)

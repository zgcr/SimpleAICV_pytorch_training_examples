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
def predict(image, select_points, mask_out_idx):
    set_seed(config.seed)

    assert len(select_points) >= 1, 'no prompt point!'

    origin_image, resized_img, scale, scaled_size, origin_size = preprocess_image(
        image, config.input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    resized_input_points = []
    resized_input_labels = []
    for per_select_points in select_points:
        resized_input_points.append([
            per_select_points[0][0],
            per_select_points[0][1],
        ])
        resized_input_labels.append([per_select_points[1]])
    resized_input_points = np.array(resized_input_points) * scale
    resized_input_labels = np.array(resized_input_labels)

    prompt_points = np.concatenate(
        [resized_input_points, resized_input_labels], axis=1)

    input_prompt_point = torch.tensor(np.expand_dims(prompt_points,
                                                     axis=0)).float().cuda()

    batch_prompts = {
        'prompt_point': input_prompt_point,
        'prompt_box': None,
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


#################################################################

# points bgr color
colors = [(0, 0, 255), (0, 255, 0)]
markers = [1, 5]

with gr.Blocks() as demo:
    with gr.Tab(label='Segment Anything!'):
        with gr.Row():
            with gr.Column():
                # store original image without points, default None
                original_image = gr.State(value=None)
                input_image = gr.Image(type='pil')
                # point prompt
                with gr.Column():
                    # store points
                    selected_points = gr.State([])
                    with gr.Row():
                        gr.Markdown(
                            'Click on the image to select prompt point.')
                        undo_point_button = gr.Button('Undo point')
                    with gr.Row():
                        gr.Markdown('Choose prompt point type.')
                        radio = gr.Radio(
                            choices=['foreground_point', 'background_point'],
                            value='foreground_point',
                            label='prompt point type')
                    with gr.Row():
                        gr.Markdown('Choose sam model mask out idx.')
                        mask_out_idx = gr.Slider(minimum=0,
                                                 maximum=3,
                                                 value=0,
                                                 step=1,
                                                 label='mask out idx')
                # run button
                run_button = gr.Button('RUN!')
            # show image with mask
            with gr.Tab(label='Image with Mask'):
                output_image_with_mask = gr.Image(type='pil')
            # only show mask
            with gr.Tab(label='Mask'):
                output_mask = gr.Image(type='pil')

    # once user upload an image, the original image is stored in `original_image`
    def store_image(origin_image):

        return origin_image, []

    # user click the image to get points, and show the points on the image
    def get_point(image, select_points, point_type, evt: gr.SelectData):
        temp_image = image.copy()

        # append the foreground_point
        if point_type == 'foreground_point':
            select_points.append((evt.index, 1))
        # append the background_point
        elif point_type == 'background_point':
            select_points.append((evt.index, 0))
        # default foreground_point
        else:
            select_points.append((evt.index, 1))

        # PIL image(RGB) to opencv image(RGB)
        temp_image = np.asarray(temp_image).astype(np.float32)
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR)

        # draw points
        for point, label in select_points:
            cv2.drawMarker(temp_image,
                           point,
                           colors[label],
                           markerType=markers[label],
                           markerSize=20,
                           thickness=5)

        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        temp_image = Image.fromarray(np.uint8(temp_image))

        return temp_image

    # undo the selected point
    def undo_points(origin_image, select_points):
        temp_image = origin_image.copy()
        # PIL image(RGB) to opencv image(RGB)
        temp_image = np.asarray(temp_image).astype(np.float32)
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR)

        # draw points
        if len(select_points) != 0:
            select_points.pop()
            for point, label in select_points:
                cv2.drawMarker(temp_image,
                               point,
                               colors[label],
                               markerType=markers[label],
                               markerSize=20,
                               thickness=5)

        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        temp_image = Image.fromarray(np.uint8(temp_image))

        return temp_image

    input_image.upload(store_image, [input_image],
                       [original_image, selected_points])

    input_image.select(get_point, [input_image, selected_points, radio],
                       [input_image])
    undo_point_button.click(undo_points, [original_image, selected_points],
                            [input_image])
    run_button.click(predict,
                     inputs=[original_image, selected_points, mask_out_idx],
                     outputs=[output_image_with_mask, output_mask])

# local website: http://127.0.0.1:6006/
demo.queue().launch(share=True,
                    server_name='0.0.0.0',
                    server_port=6006,
                    show_error=True)

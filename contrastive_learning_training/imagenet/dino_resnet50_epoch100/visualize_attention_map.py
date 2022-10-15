import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from tools.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='save reconstruction image')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()


def main():
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from visualize_config import config
    save_image_dir = os.path.join(args.work_dir, 'save_image')

    set_seed(config.seed)

    os.makedirs(save_image_dir) if not os.path.exists(save_image_dir) else None

    show_dataset = config.show_dataset
    model = config.model
    model.eval()

    with torch.no_grad():
        for i, per_sample in tqdm(enumerate(show_dataset)):
            if i >= config.save_num:
                break

            image = per_sample['image']
            image = np.array(image).astype(np.float32)
            image = np.expand_dims(image, axis=0)
            image = torch.from_numpy(image)
            # 1 H W 3 ->1 3 H W
            image = image.permute(0, 3, 1, 2)

            input_image = image
            origin_h, origin_w = input_image.shape[2], input_image.shape[3]

            output = model.get_attention_map(image.float())
            output = F.interpolate(output,
                                   size=(origin_h, origin_w),
                                   mode='bilinear',
                                   align_corners=True)
            output = output.squeeze(dim=0)
            output = torch.einsum('chw->hwc', output)
            output = torch.mean(output, dim=-1)
            output = torch.sigmoid(output)
            output = torch.unsqueeze(output, dim=-1).repeat(1, 1, 3)

            input_image = input_image.squeeze(dim=0)
            input_image = torch.einsum('chw->hwc', input_image)

            input_image = torch.clip(
                (input_image * np.array(config.std) + np.array(config.mean)) *
                255, 0, 255).int().numpy().astype(np.float32)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

            output = output.numpy().astype(np.float32)
            output = (output - np.min(output)) / (np.max(output) -
                                                  np.min(output)) * 255.
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            input_image_name = f'image_{i}_input.jpg'
            output_image_name = f'image_{i}_output.jpg'

            cv2.imencode('.jpg', input_image)[1].tofile(
                os.path.join(save_image_dir, input_image_name))
            cv2.imencode('.jpg', output)[1].tofile(
                os.path.join(save_image_dir, output_image_name))


if __name__ == '__main__':
    main()

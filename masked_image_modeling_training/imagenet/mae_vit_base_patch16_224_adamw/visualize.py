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

            output, mask = model(image.float())
            # unpatch output
            # outputs: (1, L, patch_size**2 *3)
            # image: (1, 3, H, W)
            output = model.patch_to_images(output)
            output = torch.einsum('nchw->nhwc', output)

            # (1, H*W, p*p*3)
            patch_size = model.patch_size
            mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 * 3)
            # 1 is removing, 0 is keeping
            mask = model.patch_to_images(mask)
            mask = torch.einsum('nchw->nhwc', mask)

            input_image = torch.einsum('nchw->nhwc', input_image)

            # origin_image
            origin_image = input_image.squeeze(dim=0)
            origin_image = torch.clip(
                (origin_image * np.array(config.std) + np.array(config.mean)) *
                255, 0, 255).int().numpy().astype(np.float32)
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)

            # masked image
            masked_image = (input_image * (1 - mask)).squeeze(dim=0)
            masked_image = torch.clip(
                (masked_image * np.array(config.std) + np.array(config.mean)) *
                255, 0, 255).int().numpy().astype(np.float32)
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

            # reconstruction image
            reconstruciton_image = output.squeeze(dim=0)
            reconstruciton_image = torch.clip(
                (reconstruciton_image * np.array(config.std) +
                 np.array(config.mean)) * 255, 0,
                255).int().numpy().astype(np.float32)
            reconstruciton_image = cv2.cvtColor(reconstruciton_image,
                                                cv2.COLOR_RGB2BGR)

            # reconstruction pasted with visible patches
            combined_image = (input_image * (1 - mask) +
                              output * mask).squeeze(dim=0)
            combined_image = torch.clip(
                (combined_image * np.array(config.std) + np.array(config.mean))
                * 255, 0, 255).int().numpy().astype(np.float32)
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

            origin_image_name = f'image_{i}_original.jpg'
            masked_image_name = f'image_{i}_masked.jpg'
            reconstruciton_image_name = f'image_{i}_reconstruciton.jpg'
            combined_image_name = f'image_{i}_combined.jpg'

            cv2.imencode('.jpg', origin_image)[1].tofile(
                os.path.join(save_image_dir, origin_image_name))
            cv2.imencode('.jpg', masked_image)[1].tofile(
                os.path.join(save_image_dir, masked_image_name))
            cv2.imencode('.jpg', reconstruciton_image)[1].tofile(
                os.path.join(save_image_dir, reconstruciton_image_name))
            cv2.imencode('.jpg', combined_image)[1].tofile(
                os.path.join(save_image_dir, combined_image_name))


if __name__ == '__main__':
    main()

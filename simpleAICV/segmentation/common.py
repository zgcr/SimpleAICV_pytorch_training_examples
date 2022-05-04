import cv2
import math
import numpy as np

import torch


class RetinaStyleResize:

    def __init__(self,
                 resize=400,
                 divisor=32,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=[0.8, 1.0]):
        self.resize = resize
        self.divisor = divisor
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range
        self.ratio = 1333. / 800

    def __call__(self, sample):
        image, box_targets, mask_targets, class_targets, scale, size = sample[
            'image'], sample['box_annots'], sample['mask_annots'], sample[
                'class_annots'], sample['scale'], sample['size']
        h, w, _ = image.shape

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            scales = (resize_list[random_idx],
                      int(round(self.resize * self.ratio)))
        else:
            scales = (self.resize, int(round(self.resize * self.ratio)))

        max_long_edge, max_short_edge = max(scales), min(scales)
        factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        padded_masks = np.zeros(
            (mask_targets.shape[0], resize_h + pad_h, resize_w + pad_w),
            dtype=np.float32)
        if mask_targets.shape[0] > 0:
            # [mask_num,h,w]->[h,w,mask_num]->[resize_h,resize_w,mask_num]->[mask_num,resize_h,resize_w]
            mask_targets = cv2.resize(mask_targets.transpose(1, 2, 0),
                                      (resize_w, resize_h))
            if len(mask_targets.shape) < 3:
                mask_targets = np.expand_dims(mask_targets, axis=-1)
            mask_targets = mask_targets.transpose(2, 0, 1)
            # set resize position value <=0 to 0,value >0 to 1
            mask_targets[mask_targets <= 0] = 0
            mask_targets[mask_targets > 0] = 1
            padded_masks[:, :resize_h, :resize_w] = mask_targets

        factor = np.float32(factor)
        box_targets *= factor
        scale *= factor

        return {
            'image': padded_image,
            'box_annots': box_targets,
            'mask_annots': padded_masks,
            'class_annots': class_targets,
            'scale': scale,
            'size': size,
        }


class YoloStyleResize:

    def __init__(self,
                 resize=640,
                 divisor=32,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=[0.5, 1.0]):
        self.resize = resize
        self.divisor = divisor
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range

    def __call__(self, sample):
        image, box_targets, mask_targets, class_targets, scale, size = sample[
            'image'], sample['box_annots'], sample['mask_annots'], sample[
                'class_annots'], sample['scale'], sample['size']
        h, w, _ = image.shape

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            final_resize = resize_list[random_idx]
        else:
            final_resize = self.resize

        factor = final_resize / max(h, w)

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        padded_masks = np.zeros(
            (mask_targets.shape[0], resize_h + pad_h, resize_w + pad_w),
            dtype=np.float32)
        if mask_targets.shape[0] > 0:
            # [mask_num,h,w]->[h,w,mask_num]->[resize_h,resize_w,mask_num]->[mask_num,resize_h,resize_w]
            mask_targets = cv2.resize(mask_targets.transpose(1, 2, 0),
                                      (resize_w, resize_h))
            if len(mask_targets.shape) < 3:
                mask_targets = np.expand_dims(mask_targets, axis=-1)
            mask_targets = mask_targets.transpose(2, 0, 1)
            # set resize position value <=0 to 0,value >0 to 1
            mask_targets[mask_targets <= 0] = 0
            mask_targets[mask_targets > 0] = 1
            padded_masks[:, :resize_h, :resize_w] = mask_targets

        factor = np.float32(factor)
        box_targets *= factor
        scale *= factor

        return {
            'image': padded_image,
            'box_annots': box_targets,
            'mask_annots': padded_masks,
            'class_annots': class_targets,
            'scale': scale,
            'size': size,
        }


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, box_targets, mask_targets, class_targets, scale, size = sample[
            'image'], sample['box_annots'], sample['mask_annots'], sample[
                'class_annots'], sample['scale'], sample['size']

        if box_targets.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            _, w, _ = image.shape

            x1 = box_targets[:, 0].copy()
            x2 = box_targets[:, 2].copy()

            box_targets[:, 0] = w - x2
            box_targets[:, 2] = w - x1

            mask_targets = mask_targets[:, :, ::-1]

        return {
            'image': image,
            'box_annots': box_targets,
            'mask_annots': mask_targets,
            'class_annots': class_targets,
            'scale': scale,
            'size': size,
        }


class RandomCrop:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, box_targets, mask_targets, class_targets, scale, size = sample[
            'image'], sample['box_annots'], sample['mask_annots'], sample[
                'class_annots'], sample['scale'], sample['size']

        if box_targets.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(box_targets[:, 0:2], axis=0),
                np.max(box_targets[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            crop_xmin = max(
                0, int(max_bbox[0] - np.random.uniform(0, max_left_trans)))
            crop_ymin = max(
                0, int(max_bbox[1] - np.random.uniform(0, max_up_trans)))
            crop_xmax = min(
                w, int(max_bbox[2] + np.random.uniform(0, max_right_trans)))
            crop_ymax = min(
                h, int(max_bbox[3] + np.random.uniform(0, max_down_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            box_targets[:, [0, 2]] = box_targets[:, [0, 2]] - crop_xmin
            box_targets[:, [1, 3]] = box_targets[:, [1, 3]] - crop_ymin

            mask_targets = mask_targets[:, crop_ymin:crop_ymax,
                                        crop_xmin:crop_xmax]

        return {
            'image': image,
            'box_annots': box_targets,
            'mask_annots': mask_targets,
            'class_annots': class_targets,
            'scale': scale,
            'size': size,
        }


class RandomTranslate:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, box_targets, mask_targets, class_targets, scale, size = sample[
            'image'], sample['box_annots'], sample['mask_annots'], sample[
                'class_annots'], sample['scale'], sample['size']

        if box_targets.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(box_targets[:, 0:2], axis=0),
                np.max(box_targets[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            tx = np.random.uniform(-(max_left_trans - 1),
                                   (max_right_trans - 1))
            ty = np.random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            box_targets[:, [0, 2]] = box_targets[:, [0, 2]] + tx
            box_targets[:, [1, 3]] = box_targets[:, [1, 3]] + ty

            # [mask_num,h,w]->[h,w,mask_num]->[resize_h,resize_w,mask_num]->[mask_num,resize_h,resize_w]
            mask_targets = cv2.warpAffine(mask_targets.transpose(1, 2, 0), M,
                                          (w, h))
            if len(mask_targets.shape) < 3:
                mask_targets = np.expand_dims(mask_targets, axis=-1)
            mask_targets = mask_targets.transpose(2, 0, 1)
            # set resize position value <=0 to 0,value >0 to 1
            mask_targets[mask_targets <= 0] = 0
            mask_targets[mask_targets > 0] = 1

        return {
            'image': image,
            'box_annots': box_targets,
            'mask_annots': mask_targets,
            'class_annots': class_targets,
            'scale': scale,
            'size': size,
        }


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        image, box_targets, mask_targets, class_targets, scale, size = sample[
            'image'], sample['box_annots'], sample['mask_annots'], sample[
                'class_annots'], sample['scale'], sample['size']

        image = image / 255.

        return {
            'image': image,
            'box_annots': box_targets,
            'mask_annots': mask_targets,
            'class_annots': class_targets,
            'scale': scale,
            'size': size,
        }


class InstanceSegmentationCollater:

    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, data):
        images = [s['image'] for s in data]
        box_targets = [s['box_annots'] for s in data]
        mask_targets = [s['mask_annots'] for s in data]
        class_targets = [s['class_annots'] for s in data]
        scales = [s['scale'] for s in data]
        sizes = [s['size'] for s in data]

        batch_num = len(images)
        max_h = max(image.shape[0] for image in images)
        max_w = max(image.shape[1] for image in images)

        pad_h = 0 if max_h % self.divisor == 0 else self.divisor - max_h % self.divisor
        pad_w = 0 if max_w % self.divisor == 0 else self.divisor - max_w % self.divisor

        input_images = np.zeros((batch_num, max_h + pad_h, max_w + pad_w, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)

        mask_num = max(mask_target.shape[0] for mask_target in mask_targets)
        max_h = max(mask_target.shape[1] for mask_target in mask_targets)
        max_w = max(mask_target.shape[2] for mask_target in mask_targets)

        pad_h = 0 if max_h % self.divisor == 0 else self.divisor - max_h % self.divisor
        pad_w = 0 if max_w % self.divisor == 0 else self.divisor - max_w % self.divisor

        input_masks = np.zeros(
            (batch_num, mask_num, max_h + pad_h, max_w + pad_w),
            dtype=np.float32)
        for i, (per_image_mask,
                per_image_class) in enumerate(zip(mask_targets,
                                                  class_targets)):
            assert per_image_mask.shape[0] == per_image_class.shape[0]
            per_image_mask_num, per_image_h, per_image_w = per_image_mask.shape[
                0], per_image_mask.shape[1], per_image_mask.shape[2]
            input_masks[i, 0:per_image_mask_num, 0:per_image_h,
                        0:per_image_w] = per_image_mask

        # [B,mask_num,h,w]
        input_masks = torch.from_numpy(input_masks)

        max_object_num = max(box_target.shape[0] for box_target in box_targets)
        input_boxes = np.ones(
            (len(box_targets), max_object_num, 4), dtype=np.float32) * (-1)
        input_classes = np.ones(
            (len(box_targets), max_object_num), dtype=np.float32) * (-1)
        if max_object_num > 0:
            for i, box_target in enumerate(box_targets):
                if box_target.shape[0] > 0:
                    input_boxes[i, :box_target.shape[0], :] = box_target
            for i, class_target in enumerate(class_targets):
                if class_target.shape[0] > 0:
                    input_classes[i, :class_target.shape[0]] = class_target

        input_boxes = torch.from_numpy(input_boxes)
        input_classes = torch.from_numpy(input_classes)

        scales = np.array(scales, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'box_annots': input_boxes,
            'mask_annots': input_masks,
            'class_annots': input_classes,
            'scale': scales,
            'size': sizes,
        }


class SemanticSegmentationCollater:

    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, data):
        images = [s['image'] for s in data]
        box_targets = [s['box_annots'] for s in data]
        mask_targets = [s['mask_annots'] for s in data]
        class_targets = [s['class_annots'] for s in data]
        scales = [s['scale'] for s in data]
        sizes = [s['size'] for s in data]

        max_h = max(image.shape[0] for image in images)
        max_w = max(image.shape[1] for image in images)

        pad_h = 0 if max_h % self.divisor == 0 else self.divisor - max_h % self.divisor
        pad_w = 0 if max_w % self.divisor == 0 else self.divisor - max_w % self.divisor

        input_images = np.zeros((len(images), max_h + pad_h, max_w + pad_w, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)

        max_h = max(mask_target.shape[1] for mask_target in mask_targets)
        max_w = max(mask_target.shape[2] for mask_target in mask_targets)

        pad_h = 0 if max_h % self.divisor == 0 else self.divisor - max_h % self.divisor
        pad_w = 0 if max_w % self.divisor == 0 else self.divisor - max_w % self.divisor

        input_masks = np.zeros(
            (len(mask_targets), max_h + pad_h, max_w + pad_w),
            dtype=np.float32)
        for i, (per_image_mask,
                per_image_class) in enumerate(zip(mask_targets,
                                                  class_targets)):
            assert per_image_mask.shape[0] == per_image_class.shape[0]
            per_image_h, per_image_w = per_image_mask.shape[
                1], per_image_mask.shape[2]
            final_per_image_mask = np.zeros((per_image_h, per_image_w),
                                            dtype=np.float32)
            for per_mask, per_class in zip(per_image_mask, per_image_class):
                # final semantic mask coco class label from 1 to 80
                final_per_image_mask[
                    per_mask > 0] = per_mask[per_mask > 0] * (per_class)

            input_masks[i, 0:per_image_h, 0:per_image_w] = final_per_image_mask
        # [B,h,w]
        input_masks = torch.from_numpy(input_masks)

        max_object_num = max(box_target.shape[0] for box_target in box_targets)
        input_boxes = np.ones(
            (len(box_targets), max_object_num, 4), dtype=np.float32) * (-1)
        input_classes = np.ones(
            (len(box_targets), max_object_num), dtype=np.float32) * (-1)
        if max_object_num > 0:
            for i, box_target in enumerate(box_targets):
                if box_target.shape[0] > 0:
                    input_boxes[i, :box_target.shape[0], :] = box_target
            for i, class_target in enumerate(class_targets):
                if class_target.shape[0] > 0:
                    input_classes[i, :class_target.shape[0]] = class_target

        input_boxes = torch.from_numpy(input_boxes)
        input_classes = torch.from_numpy(input_classes)

        scales = np.array(scales, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'box_annots': input_boxes,
            'mask_annots': input_masks,
            'class_annots': input_classes,
            'scale': scales,
            'size': sizes,
        }


class SegmentationDataPrefetcher:

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            images, box_targets, mask_targets, class_targets = sample[
                'image'], sample['box_annots'], sample['mask_annots'], sample[
                    'class_annots']
            self.next_images, self.next_box_targets, self.next_mask_targets, self.next_class_targets = images, box_targets, mask_targets, class_targets
        except StopIteration:
            self.next_images = None
            self.next_box_targets = None
            self.next_mask_targets = None
            self.next_class_targets = None
            return
        with torch.cuda.stream(self.stream):
            self.next_images = self.next_images.cuda(non_blocking=True)
            self.next_box_targets = self.next_box_targets.cuda(
                non_blocking=True)
            self.next_mask_targets = self.next_mask_targets.cuda(
                non_blocking=True)
            self.next_class_targets = self.next_class_targets.cuda(
                non_blocking=True)
            self.next_images = self.next_images.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        images, box_targets, mask_targets, class_targets = self.next_images, self.next_box_targets, self.next_mask_targets, self.next_class_targets
        self.preload()

        return images, box_targets, mask_targets, class_targets


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    only load layer parameters which has same layer name and same layer weight shape
    '''
    if not saved_model_path:
        print('No pretrained model file!')
        return

    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))

    filtered_state_dict = {
        name: weight
        for name, weight in saved_state_dict.items()
        if name in model.state_dict() and not any(
            excluded_name in name for excluded_name in excluded_layer_name)
        and weight.shape == model.state_dict()[name].shape
    }

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        model.load_state_dict(filtered_state_dict, strict=False)

    return
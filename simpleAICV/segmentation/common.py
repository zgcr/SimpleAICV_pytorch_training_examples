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
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, origin_hw = sample['image'], sample[
            'annots'], sample['scale'], sample['origin_hw']
        boxes, masks, classes = annots['box'], annots['mask'], annots['class']
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
                      math.ceil(self.resize * self.ratio))
        else:
            scales = (self.resize, math.ceil(self.resize * self.ratio))

        max_long_edge, max_short_edge = max(scales), min(scales)
        factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))

        resize_h, resize_w = math.ceil(h * factor), math.ceil(w * factor)
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        masks = [
            cv2.resize(((masks[i:i + 1]).transpose((1, 2, 0))),
                       (resize_w, resize_h)) for i in range(masks.shape[0])
        ]
        masks = np.array(masks, dtype=np.float32)

        padded_masks = np.zeros(
            (masks.shape[0], resize_h + pad_h, resize_w + pad_w),
            dtype=np.float32)
        padded_masks[:, :resize_h, :resize_w] = masks

        factor = np.float32(factor)
        boxes *= factor
        scale *= factor

        annots = {
            'box': boxes,
            'mask': padded_masks,
            'class': classes,
        }

        return {
            'image': padded_image,
            'annots': annots,
            'scale': scale,
            'origin_hw': origin_hw
        }


class YoloStyleResize:
    def __init__(self,
                 resize=600,
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
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, origin_hw = sample['image'], sample[
            'annots'], sample['scale'], sample['origin_hw']
        boxes, masks, classes = annots['box'], annots['mask'], annots['class']
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

        resize_h, resize_w = math.ceil(h * factor), math.ceil(w * factor)
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        masks = [
            cv2.resize(((masks[i:i + 1]).transpose((1, 2, 0))),
                       (resize_w, resize_h)) for i in range(masks.shape[0])
        ]
        masks = np.array(masks, dtype=np.float32)

        padded_masks = np.zeros(
            (masks.shape[0], resize_h + pad_h, resize_w + pad_w),
            dtype=np.float32)
        padded_masks[:, :resize_h, :resize_w] = masks

        factor = np.float32(factor)
        boxes *= factor
        scale *= factor

        annots = {
            'box': boxes,
            'mask': padded_masks,
            'class': classes,
        }

        return {
            'image': padded_image,
            'annots': annots,
            'scale': scale,
            'origin_hw': origin_hw
        }


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, origin_hw = sample['image'], sample[
            'annots'], sample['scale'], sample['origin_hw']
        boxes, masks, classes = annots['box'], annots['mask'], annots['class']

        if classes.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.flip_prob:
            image = image[:, ::-1, :]
            _, w, _ = image.shape

            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()

            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1

            masks = masks[:, :, ::-1]

        annots = {
            'box': boxes,
            'mask': masks,
            'class': classes,
        }

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'origin_hw': origin_hw
        }


class RandomCrop:
    def __init__(self, crop_prob=0.5):
        self.crop_prob = crop_prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, origin_hw = sample['image'], sample[
            'annots'], sample['scale'], sample['origin_hw']
        boxes, masks, classes = annots['box'], annots['mask'], annots['class']

        if classes.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.crop_prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [np.min(boxes[:, 0:2], axis=0),
                 np.max(boxes[:, 2:4], axis=0)],
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

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
            masks = masks[:, crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_xmin
            boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_ymin

        annots = {
            'box': boxes,
            'mask': masks,
            'class': classes,
        }

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'origin_hw': origin_hw
        }


class RandomTranslate:
    def __init__(self, translate_prob=0.5):
        self.translate_prob = translate_prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, origin_hw = sample['image'], sample[
            'annots'], sample['scale'], sample['origin_hw']
        boxes, masks, classes = annots['box'], annots['mask'], annots['class']

        if classes.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.translate_prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [np.min(boxes[:, 0:2], axis=0),
                 np.max(boxes[:, 2:4], axis=0)],
                axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            tx = np.random.uniform(-(max_left_trans - 1),
                                   (max_right_trans - 1))
            ty = np.random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            boxes[:, [0, 2]] = boxes[:, [0, 2]] + tx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] + ty

            masks = [
                cv2.warpAffine(((masks[i:i + 1]).transpose((1, 2, 0))), M,
                               (w, h)) for i in range(masks.shape[0])
            ]
            masks = np.array(masks, dtype=np.float32)

        annots = {
            'box': boxes,
            'mask': masks,
            'class': classes,
        }

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'origin_hw': origin_hw
        }


class Normalize:
    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, origin_hw = sample['image'], sample[
            'annots'], sample['scale'], sample['origin_hw']

        image = image / 255.

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'origin_hw': origin_hw
        }


class SegmentationCollater:
    def __init__(self):
        pass

    def next(self, data):
        images = [s['image'] for s in data]
        annots = [s['annots'] for s in data]
        scales = [s['scale'] for s in data]
        origin_hws = [s['origin_hw'] for s in data]

        boxes, masks, classes = [], [], []
        for s in data:
            boxes.append(s['annots']['box'])
            masks.append(s['annots']['mask'])
            classes.append(s['annots']['class'])

        max_h = max(image.shape[0] for image in images)
        max_w = max(image.shape[1] for image in images)

        input_images = np.zeros((len(images), max_h, max_w, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # input_images shape:[B,H,W,3] ->[B,3,H,W]
        input_images = input_images.permute(0, 3, 1, 2)

        max_object_num = max(box.shape[0] for box in boxes)

        if max_object_num > 0:
            input_boxes = np.ones(
                (len(boxes), max_object_num, 4), dtype=np.float32) * (-1)
            for i, box in enumerate(boxes):
                if box.shape[0] > 0:
                    input_boxes[i, :box.shape[0], :] = box
            input_classes = np.ones(
                (len(classes), max_object_num, 1), dtype=np.float32) * (-1)
            for i, per_class in enumerate(classes):
                if per_class.shape[0] > 0:
                    input_classes[i, :per_class.shape[0], :] = per_class
            input_masks = np.zeros((len(masks), max_object_num, max_h, max_w),
                                   dtype=np.float32)
            for i, per_mask in enumerate(masks):
                if per_mask.shape[0] > 0:
                    input_masks[i, :per_mask.shape[0], 0:per_mask.shape[1],
                                0:per_mask.shape[2]] = per_mask
        else:
            input_boxes = np.ones((len(boxes), 1, 4), dtype=np.float32) * (-1)
            input_classes = np.ones(
                (len(classes), 1, 1), dtype=np.float32) * (-1)
            input_masks = np.zeros((len(masks), 1, max_h, max_w),
                                   dtype=np.float32)

        input_boxes = torch.from_numpy(input_boxes)
        input_classes = torch.from_numpy(input_classes)
        input_masks = torch.from_numpy(input_masks)
        scales = np.array(scales, dtype=np.float32)
        scales = torch.from_numpy(scales)
        origin_hws = np.array(origin_hws, dtype=np.float32)
        origin_hws = torch.from_numpy(origin_hws)

        input_annots = {
            'box': input_boxes,
            'mask': input_masks,
            'class': input_classes,
        }

        return {
            'image': input_images,
            'annots': input_annots,
            'scale': scales,
            'origin_hw': origin_hws,
        }


class SegmentationDataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            images, annots = sample['image'], sample['annots']
            boxes, masks, classes = annots['box'], annots['mask'], annots[
                'class']
            self.next_images, self.next_boxes, self.next_masks, self.next_classes = images, boxes, masks, classes
            self.next_input, self.next_annot = sample['image'], sample[
                'annots']
        except StopIteration:
            self.next_images = None
            self.next_boxes = None
            self.next_masks = None
            self.next_classes = None
            return
        with torch.cuda.stream(self.stream):
            self.next_images = self.next_images.cuda(non_blocking=True)
            self.next_boxes = self.next_boxes.cuda(non_blocking=True)
            self.next_masks = self.next_masks.cuda(non_blocking=True)
            self.next_classes = self.next_classes.cuda(non_blocking=True)
            self.next_images = self.next_images.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        images = self.next_images
        boxes = self.next_boxes
        masks = self.next_masks
        classes = self.next_classes
        self.preload()

        return images, boxes, masks, classes


def load_state_dict(saved_state_dict, model, excluded_layer_name=()):
    '''
    saved_state_dict: a saved model.state_dict()
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    only load layer parameters which has same layer name and same layer weight shape
    '''
    filtered_state_dict = {
        name: weight
        for name, weight in saved_state_dict.items()
        if name in model.state_dict() and not any(
            excluded_name in name for excluded_name in excluded_layer_name)
        and weight.shape == model.state_dict()[name].shape
    }

    model.load_state_dict(filtered_state_dict, strict=False)

    return
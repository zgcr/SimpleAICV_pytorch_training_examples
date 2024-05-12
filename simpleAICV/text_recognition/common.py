import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import json
import math
import numpy as np

import torch

from simpleAICV.classification.common import load_state_dict, AverageMeter


class RandomScale:

    def __init__(self, scale=[0.9, 1.0], prob=0.5):
        self.scale = scale
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            return sample

        image, label, scale, size = sample['image'], sample['label'], sample[
            'scale'], sample['size']

        h, w = image.shape[0], image.shape[1]

        center = (int(w / 2), int(h / 2))
        scale = np.random.uniform(self.scale[0], self.scale[1])
        M = cv2.getRotationMatrix2D(center, 0, scale)

        radian = np.deg2rad(0)
        new_w = int((abs(np.sin(radian) * h) + abs(np.cos(radian) * w)))
        new_h = int((abs(np.cos(radian) * h) + abs(np.sin(radian) * w)))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        image = cv2.warpAffine(image,
                               M, (new_w, new_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        return sample


class RandomGaussianBlur:

    def __init__(self, sigma=[0.5, 1.5], prob=0.5):
        self.sigma = sigma
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            return sample

        image, label, scale, size = sample['image'], sample['label'], sample[
            'scale'], sample['size']

        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        ksize = int(2 * ((sigma - 0.8) / 0.3 + 1) + 1)
        if ksize % 2 == 0:
            ksize += 1

        image = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        return sample


class RandomBrightness:

    def __init__(self, brightness=[0.5, 1.5], prob=0.3):
        self.brightness = brightness
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            return sample

        image, label, scale, size = sample['image'], sample['label'], sample[
            'scale'], sample['size']

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        brightness = np.random.uniform(self.brightness[0], self.brightness[1])

        mask = hsv_image[:, :, 2] * brightness > 255
        hsv_channel = np.where(mask, 255, hsv_image[:, :, 2] * brightness)
        hsv_image[:, :, 2] = hsv_channel
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        return sample


class RandomRotate:

    def __init__(self, angle=[-5, 5], prob=0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            return sample

        image, label, scale, size = sample['image'], sample['label'], sample[
            'scale'], sample['size']

        matrix = np.eye(3, dtype=np.float32)
        h, w = image.shape[0], image.shape[1]
        center_matrix = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]],
                                 dtype=np.float32)

        matrix = np.matmul(center_matrix, matrix)
        angle = np.random.uniform(self.angle[0], self.angle[1])
        rad = -1.0 * np.deg2rad(angle)
        rad_matrix = np.array([[np.cos(rad), np.sin(rad), 0],
                               [-np.sin(rad), np.cos(rad), 0], [0, 0, 1]],
                              dtype=np.float32)
        matrix = np.matmul(rad_matrix, matrix)

        invert_center_matrix = np.array(
            [[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], dtype=np.float32)
        matrix = np.matmul(invert_center_matrix, matrix)

        corners_matrix = np.array([[0, 0], [0, h - 1], [w - 1, h - 1],
                                   [w - 1, 0]])
        x, y = np.transpose(corners_matrix)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = np.dot(src.T, matrix.T)
        dst[dst[:, 2] == 0, 2] = np.finfo(float).eps
        dst[:, :2] /= dst[:, 2:3]
        corners_matrix = dst[:, :2]

        minc, minr, maxc, maxr = corners_matrix[:, 0].min(
        ), corners_matrix[:, 1].min(), corners_matrix[:, 0].max(
        ), corners_matrix[:, 1].max()
        new_h, new_w = int(np.round(maxr - minr + 1)), int(
            np.round(maxc - minc + 1))

        translate_matrix = np.array([[1, 0, -minc], [0, 1, -minr], [0, 0, 1]])
        matrix = np.matmul(translate_matrix, matrix)

        image = cv2.warpAffine(image,
                               matrix[0:2, :], (new_w, new_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        return sample


class WarpMLS:

    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    def __bilinear_interp(self, x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 *
                                                      (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1. / ((i - self.dst_pts[k][0]) *
                                 (i - self.dst_pts[k][0]) +
                                 (j - self.dst_pts[k][1]) *
                                 (j - self.dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - \
                                    np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + \
                                    np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdx[i, j], self.rdx[i,
                                                                          nj],
                                                 self.rdx[ni, j], self.rdx[ni,
                                                                           nj])
                delta_y = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdy[i, j], self.rdy[i,
                                                                          nj],
                                                 self.rdy[ni, j], self.rdy[ni,
                                                                           nj])
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h,
                    j:j + w] = self.__bilinear_interp(x, y, self.src[nyi, nxi],
                                                      self.src[nyi, nxi1],
                                                      self.src[nyi1, nxi],
                                                      self.src[nyi1, nxi1])

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.float32)

        return dst


class Distort:
    """
    https://github.com/RubanSeven/Text-Image-Augmentation-python
    """

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            return sample

        image, label, scale, size = sample['image'], sample['label'], sample[
            'scale'], sample['size']

        img_h, img_w = image.shape[0], image.shape[1]

        if img_w < 3 * img_h:
            return sample

        if len(label) < 3:
            return sample

        max_segment, segment = len(label), 0
        if max_segment < 20:
            segment = max_segment
        elif 20 < max_segment < 40:
            segment = int(max_segment // 2)
        else:
            segment = int(max_segment // 4)

        cut = img_w // segment
        thresh = cut // 3

        if thresh <= 0:
            return sample

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append(
            [img_w - np.random.randint(thresh),
             np.random.randint(thresh)])
        dst_pts.append([
            img_w - np.random.randint(thresh),
            img_h - np.random.randint(thresh)
        ])
        dst_pts.append(
            [np.random.randint(thresh), img_h - np.random.randint(thresh)])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([
                cut * cut_idx + np.random.randint(thresh) - half_thresh,
                np.random.randint(thresh) - half_thresh
            ])
            dst_pts.append([
                cut * cut_idx + np.random.randint(thresh) - half_thresh,
                img_h + np.random.randint(thresh) - half_thresh
            ])

        trans = WarpMLS(image, src_pts, dst_pts, img_w, img_h)
        image = trans.generate()

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        return sample


class Stretch:
    """
    https://github.com/RubanSeven/Text-Image-Augmentation-python
    """

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            return sample

        image, label, scale, size = sample['image'], sample['label'], sample[
            'scale'], sample['size']

        img_h, img_w = image.shape[0], image.shape[1]

        if img_w < 3 * img_h:
            return sample

        if len(label) < 3:
            return sample

        max_segment, segment = len(label), 0
        if max_segment < 20:
            segment = max_segment
        elif 20 < max_segment < 40:
            segment = int(max_segment // 2)
        else:
            segment = int(max_segment // 4)

        cut = img_w // segment
        thresh = cut * 4 // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            move = np.random.randint(
                thresh) - half_thresh if thresh != 0 else 0
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])

        trans = WarpMLS(image, src_pts, dst_pts, img_w, img_h)
        image = trans.generate()

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        return sample


class Perspective:
    """
    https://github.com/RubanSeven/Text-Image-Augmentation-python
    """

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            return sample

        image, label, scale, size = sample['image'], sample['label'], sample[
            'scale'], sample['size']

        img_h, img_w = image.shape[0], image.shape[1]

        if img_w < 3 * img_h:
            return sample

        if len(label) < 3:
            return sample

        thresh = img_h // 2

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])

        trans = WarpMLS(image, src_pts, dst_pts, img_w, img_h)
        image = trans.generate()

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        return sample


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, scale, size = sample['image'], sample['label'], sample[
            'scale'], sample['size']

        image = image / 255.
        image = image.astype(np.float32)

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        return sample


class KeepRatioResizeTextRecognitionCollater:

    def __init__(self, resize_h=32):
        self.resize_h = resize_h

    def __call__(self, data):
        images = [s['image'] for s in data]
        labels = [s['label'] for s in data]
        scales = [s['scale'] for s in data]
        sizes = [s['size'] for s in data]

        ratios = [image.shape[1] / float(image.shape[0]) for image in images]
        max_w = int(math.floor(max(ratios) * self.resize_h))
        max_w = int(((max_w // 32) + 1) * 32)

        input_images = np.zeros((len(images), self.resize_h, max_w, 1),
                                dtype=np.float32)

        for i, image in enumerate(images):
            image = cv2.resize(image, (max(
                1, int(math.floor(self.resize_h * ratios[i]))), self.resize_h))
            image = np.expand_dims(image, 2)
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)

        return {
            'image': input_images,
            'label': labels,
            'scale': scales,
            'size': sizes,
        }


class CTCTextLabelConverter:
    """
    Convert between text label and text index
    """

    def __init__(self, chars_set_list, str_max_length=80, garbage_char='㍿'):
        """
        chars_set: set of the possible characters
        str_max_length: max length of text label in the batch
        """

        print(f"Char Num:{len(chars_set_list)}")

        self.ctc_chars_set = chars_set_list + ['[CTCblank]']
        self.ctc_chars_dict = {
            char: i
            for i, char in enumerate(self.ctc_chars_set)
        }

        # dummy '[CTCblank]' token for CTCLoss (index=self.num_classes-1)
        self.blank_index = self.ctc_chars_dict['[CTCblank]']
        self.str_max_length = str_max_length
        self.garbage_char = garbage_char

        self.num_classes = len(self.ctc_chars_set)

    def encode(self, text):
        """
        convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: text index for CTCLoss. [batch_size, str_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding CTC blank index would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), self.str_max_length).fill_(
            self.blank_index)
        for i, t in enumerate(text):
            text = list(t)
            tmp = []
            for char in text:
                if char not in self.ctc_chars_dict:
                    # garbage class index = self.num_classes
                    tmp.append(self.num_classes)
                else:
                    tmp.append(self.ctc_chars_dict[char])
            batch_text[i][:len(tmp)] = torch.LongTensor(tmp)

        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        """
        convert text-index into text-label.
        """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if t[i] == self.num_classes:
                    char_list.append(self.garbage_char)
                if t[i] < self.num_classes - 1 and (
                        not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.ctc_chars_set[t[i]])
            text = ''.join(char_list)
            texts.append(text)

        return texts


class AttenTextLabelConverter:
    """
    Convert between text label and text index
    """

    def __init__(self, chars_set_list, str_max_length=80, garbage_char='㍿'):
        """
        chars_set: set of the possible characters
        str_max_length: max length of text label in the batch
        """

        print(f"Char Num:{len(chars_set_list)}")

        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.atten_chars_set = ['[GO]', '[s]'] + chars_set_list
        self.atten_chars_dict = {
            char: i
            for i, char in enumerate(self.atten_chars_set)
        }

        self.go_index = self.atten_chars_dict['[GO]']
        self.str_max_length = str_max_length
        self.num_classes = len(self.atten_chars_set)
        self.garbage_char = garbage_char

    def encode(self, text):
        """ 
        convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token. text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        # +1 for [s] at end of sentence.
        length = [len(s) + 1 for s in text]
        str_max_length = self.str_max_length + 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text),
                                      str_max_length + 1).fill_(self.go_index)
        for i, t in enumerate(text):
            text = list(t)
            text = ['[GO]'] + text + ['[s]']

            tmp = []
            for char in text:
                if char not in self.atten_chars_dict:
                    # garbage class index = self.num_classes
                    tmp.append(self.num_classes)
                else:
                    tmp.append(self.atten_chars_dict[char])
            batch_text[i][0:len(tmp)] = torch.LongTensor(tmp)

        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        """
        convert text-index into text-label.
        """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if self.atten_chars_set[t[i]] == '[s]':
                    break
                if t[i] == self.num_classes:
                    char_list.append(self.garbage_char)
                if self.atten_chars_set[t[i]] != '[GO]':
                    char_list.append(self.atten_chars_set[t[i]])
            text = ''.join(char_list)
            texts.append(text)

        return texts


if __name__ == '__main__':
    from simpleAICV.text_recognition.char_sets.final_char_table import final_char_table

    converter = CTCTextLabelConverter(final_char_table,
                                      str_max_length=80,
                                      garbage_char='㍿')
    print("1111", converter.num_classes)

    converter = AttenTextLabelConverter(final_char_table,
                                        str_max_length=80,
                                        garbage_char='㍿')
    print("2222", converter.num_classes)

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ImageNet1KPolicy = [
    SubAugmentPolicy(0.8, "equalize", 1, 0.8, "sheary", 4),
    SubAugmentPolicy(0.4, "color", 9, 0.6, "equalize", 3),
    SubAugmentPolicy(0.4, "color", 1, 0.6, "rotate", 8),
    SubAugmentPolicy(0.8, "solarize", 3, 0.4, "equalize", 7),
    SubAugmentPolicy(0.4, "solarize", 2, 0.6, "solarize", 2),
    SubAugmentPolicy(0.2, "color", 0, 0.8, "equalize", 8),
    SubAugmentPolicy(0.4, "equalize", 8, 0.8, "solarizeadd", 3),
    SubAugmentPolicy(0.2, "shearx", 9, 0.6, "rotate", 8),
    SubAugmentPolicy(0.6, "color", 1, 1.0, "equalize", 2),
    SubAugmentPolicy(0.4, "invert", 9, 0.6, "rotate", 0),
    SubAugmentPolicy(1.0, "equalize", 9, 0.6, "sheary", 3),
    SubAugmentPolicy(0.4, "color", 7, 0.6, "equalize", 0),
    SubAugmentPolicy(0.4, "posterize", 6, 0.4, "autocontrast", 7),
    SubAugmentPolicy(0.6, "solarize", 8, 0.6, "color", 9),
    SubAugmentPolicy(0.2, "solarize", 4, 0.8, "rotate", 9),
    SubAugmentPolicy(1.0, "rotate", 7, 0.8, "translatey", 9),
    SubAugmentPolicy(0.0, "shearx", 0, 0.8, "solarize", 4),
    SubAugmentPolicy(0.8, "sheary", 0, 0.6, "color", 4),
    SubAugmentPolicy(1.0, "color", 0, 0.6, "rotate", 2),
    SubAugmentPolicy(0.8, "equalize", 4, 0.0, "equalize", 8),
    SubAugmentPolicy(1.0, "equalize", 4, 0.6, "autocontrast", 2),
    SubAugmentPolicy(0.4, "sheary", 7, 0.6, "solarizeadd", 7),
    SubAugmentPolicy(0.8, "posterize", 2, 0.6, "solarize", 10),
    SubAugmentPolicy(0.6, "solarize", 8, 0.6, "equalize", 1),
    SubAugmentPolicy(0.8, "color", 6, 0.4, "rotate", 5),
]


class Cutout:
    def __init__(self, zero_mask_size=16):
        self.zero_mask_size = int(zero_mask_size)

    def __call__(self, img):
        w, h = img.size
        height_loc, width_loc = np.random.randint(
            low=0, high=h), np.random.randint(low=0, high=w)

        lower_left = (min(h, height_loc + self.zero_mask_size // 2),
                      min(w, width_loc + self.zero_mask_size // 2))
        upper_right = (max(0, height_loc - self.zero_mask_size // 2),
                       max(0, width_loc - self.zero_mask_size // 2))

        img_pixels = img.load()
        for i in range(upper_right[0], lower_left[0]):
            # for each col
            for j in range(upper_right[1], lower_left[1]):
                # For each row
                img_pixels[i, j] = (128, 128, 128, 0)

        return img


class SubAugmentPolicy:
    def __init__(self,
                 augment_name,
                 augment_magnitude_idx,
                 prob,
                 fillcolor=(128, 128, 128)):
        self.augment_ranges = {
            'autocontrast':
            lambda level: (),
            'equalize':
            lambda level: (),
            'invert':
            lambda level: (),
            'rotate':
            lambda level: (level / 10) * 30.
            if np.random.uniform(0, 1) < 0.5 else -(level / 10) * 30.,
            'posterize':
            lambda level: (int((level / 10) * 4), ),
            'solarize':
            lambda level: (int((level / 10) * 256), ),
            'solarizeadd':
            lambda level: (int((level / 10) * 110), ),
            'color':
            lambda level: ((level / 10) * 1.8 + 0.1, ),
            'contrast':
            lambda level: ((level / 10) * 1.8 + 0.1, ),
            'brightness':
            lambda level: ((level / 10) * 1.8 + 0.1, ),
            'sharpness':
            lambda level: ((level / 10) * 1.8 + 0.1, ),
            'shearx':
            lambda level: (level / 10) * 0.3
            if np.random.uniform(0, 1) < 0.5 else -(level / 10) * 0.3,
            'sheary':
            lambda level: (level / 10) * 0.3
            if np.random.uniform(0, 1) < 0.5 else -(level / 10) * 0.3,
            'cutout':
            lambda level: (int((level / 10) * 40), ),
            'translatex':
            lambda level: (level / 10) * float(100)
            if np.random.uniform(0, 1) < 0.5 else -(level / 10) * float(100),
            'translatey':
            lambda level: (level / 10) * float(100)
            if np.random.uniform(0, 1) < 0.5 else -(level / 10) * float(100),
        }

        self.augment_funcs = {
            "autocontrast":
            lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize":
            lambda img, magnitude: ImageOps.equalize(img),
            "invert":
            lambda img, magnitude: ImageOps.invert(img),
            "rotate":
            lambda img, magnitude: img.rotate(magnitude),
            "posterize":
            lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize":
            lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "solarizeadd":
            lambda img, magnitude: ImageOps.solarize(img + 0, magnitude),
            "color":
            lambda img, magnitude: ImageEnhance.Color(img).enhance(1 +
                                                                   magnitude),
            "contrast":
            lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude),
            "brightness":
            lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude),
            "sharpness":
            lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude),
            "shearx":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude, 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "sheary":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "cutout":
            lambda img, magnitude: Cutout(magnitude)(img),
            "translatex":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0], 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translatey":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1]),
                fillcolor=fillcolor,
            ),
        }

        self.augment_name = augment_name
        self.augment_magnitude_idx = augment_magnitude_idx

        self.operation = self.augment_funcs[self.augment_name]
        self.magnitude = self.augment_ranges[self.augment_name][
            self.augment_magnitude_idx]
        self.prob = prob

    def rotate_with_fill(self, img, magnitude):
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128, ) * 4),
                               rot).convert(img.mode)

    def __call__(self, img):
        if np.random.uniform(0, 1) < self.prob:
            img = self.operation(img, self.magnitude)

        return img


class RandAugment:
    def __init__(self, N=2, M=10):
        self.transform_list = [
            'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
            'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout',
            'SolarizeAdd'
        ]
        self.N = N
        self.M = M

        def __call__(self, img):
            for i in range(self.N):
                policy_idx = np.random.randint(0, len(self.transform_list))
                policy_name = self.transform_list[policy_idx]
                magnitude_level = float(self.M)
                prob = np.random.uniform(0.2, 0.8)
                func = SubAugmentPolicy(policy_name, magnitude_level, prob)
                img = func(img)

            return img

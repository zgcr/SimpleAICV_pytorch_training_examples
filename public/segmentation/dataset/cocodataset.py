from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import math
import random
import scipy.misc
import skimage.io
import numpy as np
import skimage.color
import torch
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from lib.pycocotools.coco import COCO
from lib.pycocotools.cocoeval import COCOeval
from lib.pycocotools import mask as maskUtils
import torch.utils.data as data
from config import Config
from preprocess.InputProcess import resize_image, resize_mask, extract_bboxes, minimize_mask, compose_image_meta


class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
    """
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {
            "{}.{}".format(info['source'], info['id']): id
            for info, id in zip(self.class_info, self.class_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    #@property
    #def image_ids(self):
    #    return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        return image.transpose(2, 0, 1)


class CocoDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 subset,
                 config,
                 class_ids=None,
                 class_map=None,
                 return_coco=False,
                 argument=True,
                 use_mini_mask=False,
                 transform=None):
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.load_coco(dataset_dir, subset, class_ids, class_map, return_coco)
        self.prepare()
        self.config = config
        self.argument = argument
        self.use_mini_mask = use_mini_mask
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, img_id):
        """Load and return ground truth data for an image (image, mask, bounding boxes).
        augment: If true, apply random image augmentation. Currently, only
            horizontal flipping is offered.
        use_mini_mask: If False, returns full-size masks that are the same height
            and width as the original image. These can be big, for example
            1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
            224x224 and are generated by extracting the bounding box of the
            object and resizing it to MINI_MASK_SHAPE.
        Returns:
        image: [height, width, 3]
        shape: the original shape of the image before resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are those
            of the image unless use_mini_mask is True, in which case they are
            defined in MINI_MASK_SHAPE.
        """

        image = self.load_image(img_id)
        mask, class_ids = self.load_mask(img_id)
        shape = image.shape
        image,window,scale,padding = resize_image(image,min_dim=self.config.IMAGE_MIN_DIM,max_dim=self.config.IMAGE_MAX_DIM, \
                                                  padding=self.config.IMAGE_PADDING)
        mask = resize_mask(mask, scale, padding)
        if self.argument:
            if random.randint(0, 1):
                image = np.fliplr(image)
                mask = np.fliplr(mask)
        if self.transform:
            image = self.transform(image)
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = extract_bboxes(mask)

        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.zeros([self.num_classes], dtype=np.int32)
        source_class_ids = self.source_class_ids[self.image_info[img_id]
                                                 ["source"]]
        active_class_ids[source_class_ids] = 1

        # Resize masks to smaller size to reduce memory usage
        if self.use_mini_mask:
            mask = minimize_mask(bbox, mask, self.config.MINI_MASK_SHAPE)

        # Image meta data
        image_meta = compose_image_meta(img_id, shape, window,
                                        active_class_ids)

        return image, image_meta, class_ids, bbox, mask

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)

        self.class_from_source_map = {
            "{}.{}".format(info['source'], info['id']): id
            for info, id in zip(self.class_info, self.class_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    #@property
    #def image_ids(self):
    #    return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_coco(self,
                  dataset_dir,
                  subsets,
                  class_ids=None,
                  class_map=None,
                  return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        # Path
        self._view_map = {
            'minival2014': 'val2014',  # 5k val2014 subset
            'valminusminival2014': 'val2014',  # val2014 \setminus minival2014
            'test-dev2015': 'test2015',
        }
        self.image_ids = []
        for subset in subsets:

            image_dir = os.path.join(
                dataset_dir, 'images',
                "train2014" if subset == "train" else "val2014")

            # Create COCO object
            json_path_dict = {
                "train": "annotations/instances_train2014.json",
                "val": "annotations/instances_val2014.json",
                "minival": "annotations/instances_minival2014.json",
                "val35k": "annotations/instances_valminusminival2014.json",
            }
            coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))
            image_ids = []
            # Load all classes or a subset?
            if not class_ids:
                # All classes
                class_ids = sorted(coco.getCatIds())

            # All images or a subset?
            if class_ids:
                for id in class_ids:
                    image_ids.extend(list(coco.getImgIds(catIds=[id])))
                # Remove duplicates
                image_ids = list(set(image_ids))
            else:
                # All images
                image_ids = list(coco.imgs.keys())

            # Add classes
            for i in class_ids:
                self.add_class("coco", i, coco.loadCats(i)[0]["name"])

            # Add images
            self.image_ids += image_ids
            for i in image_ids:
                self.add_image("coco",
                               image_id=i,
                               path=os.path.join(image_dir,
                                                 coco.imgs[i]['file_name']),
                               width=coco.imgs[i]["width"],
                               height=coco.imgs[i]["height"],
                               annotations=coco.loadAnns(
                                   coco.getAnnIds(imgIds=[i],
                                                  catIds=class_ids,
                                                  iscrowd=None)))
            print(subset)
            print(len(self.image_info))
        if return_coco:
            return coco

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id("coco.{}".format(
                annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[
                            1] != image_info["width"]:
                        m = np.ones(
                            [image_info["height"], image_info["width"]],
                            dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(self, image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def det_mask_collate(batch):
    imgs = []
    dets = []
    masks = []
    image_metas = []
    class_ids = []

    for image, image_meta, class_id, bbox, mask in batch:
        imgs.append(torch.from_numpy(np.flip(image, axis=0).copy()))
        image_metas.append(image_meta)
        class_ids.append(class_id)
        dets.append(bbox)
        masks.append(mask)
    return (torch.stack(imgs, 0), image_metas, class_ids, dets, masks)


if __name__ == '__main__':
    from config import Config
    config = Config()
    coco_dataset = CocoDataset(
        dataset_dir='/mnt/lvmhdd1/zuoxin/dataset/MSCOCO',
        subset=['train', 'minival'],
        config=config)
    print('load coco dataset')
    #image, image_meta, class_ids, bbox, mask = coco_dataset.__getitem__(100)
    coco_loader = data.DataLoader(coco_dataset,
                                  4,
                                  shuffle=True,
                                  num_workers=2,
                                  collate_fn=det_mask_collate)
    coco_loader = iter(coco_loader)
    for i in range(1, 10):
        image, image_meta, class_ids, bbox, mask = next(coco_loader)
        print('image shape', image.shape)
        print('image_meta', image_meta[0].shape)
        print('class_ids', class_ids[0].shape)
        print('bbox', bbox[0].shape)
        print('mask', mask[0].shape)

    #import cv2
    #for i in range(bbox.shape[0]):
    #    image = cv2.rectangle(image,(bbox[i][1],bbox[i][0]),(bbox[i][3],bbox[i][2]),(255,0,0))
    #cv2.imwrite('img.png',image)
    #cv2.imwrite('mask.png',mask[:,:,1]*255)
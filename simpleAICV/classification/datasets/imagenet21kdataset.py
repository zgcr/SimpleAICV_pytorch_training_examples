import os
import cv2
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

filter_image_name_list = [
    'n02088094_294.JPEG',
]


class ImageNet21KSingleLabelDataset(Dataset):
    '''
    ImageNet21K Dataset:https://image-net.org/ 
    '''

    def __init__(self, root_dir, set_name='train', transform=None):
        assert set_name in ['train', 'val'], 'Wrong set name!'
        # make sure all directories in set_dir directory are sub-categories directory and no other files
        set_dir = os.path.join(root_dir, set_name)

        sub_class_name_list = []
        for per_sub_class_name in os.listdir(set_dir):
            sub_class_name_list.append(per_sub_class_name)
        sub_class_name_list = sorted(sub_class_name_list)

        self.image_path_list = []
        for per_sub_class_name in tqdm(sub_class_name_list):
            per_sub_class_dir = os.path.join(set_dir, per_sub_class_name)
            for per_image_name in os.listdir(per_sub_class_dir):
                if per_image_name in filter_image_name_list:
                    continue
                per_image_path = os.path.join(per_sub_class_dir,
                                              per_image_name)
                self.image_path_list.append(per_image_path)

        self.class_name_to_label = {
            sub_class_name: i
            for i, sub_class_name in enumerate(sub_class_name_list)
        }

        self.label_to_class_name = {
            i: sub_class_name
            for i, sub_class_name in enumerate(sub_class_name_list)
        }

        self.transform = transform

        print(f'Dataset Size:{len(self.image_path_list)}')
        print(f'Dataset Class Num:{len(self.class_name_to_label)}')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.load_label(idx)

        sample = {
            'image': image,
            'label': label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, idx):
        label = self.class_name_to_label[self.image_path_list[idx].split('/')
                                         [-2]]
        label = np.array(label)

        return label.astype(np.float32)


class ImageNet21KSemanticTreeLabelDataset(Dataset):
    '''
    ImageNet21K Dataset:https://image-net.org/ 
    '''

    def __init__(self, root_dir, set_name='train', transform=None):
        assert set_name in ['train', 'val'], 'Wrong set name!'
        # make sure all directories in set_dir directory are sub-categories directory and no other files
        semantic_tree_path = os.path.join(root_dir,
                                          'imagenet21k_miil_tree.pth')
        semantic_tree = torch.load(semantic_tree_path)

        # semantic tree key:'class_list', 'child_2_parent', 'class_tree_list', 'class_description'
        class_file_names = np.array(list(semantic_tree['class_list']))
        class_names = np.array([
            semantic_tree['class_description'][per_class_file_name]
            for per_class_file_name in class_file_names
        ])

        self.class_file_name_to_single_label = {
            per_class_file_name: i
            for i, per_class_file_name in enumerate(class_file_names)
        }

        self.single_label_to_class_file_name = {
            i: per_class_file_name
            for i, per_class_file_name in enumerate(class_file_names)
        }

        self.class_tree_list = semantic_tree['class_tree_list']
        num_classes = len(self.class_tree_list)

        self.class_depth = torch.zeros(num_classes)
        for i in range(num_classes):
            self.class_depth[i] = len(self.class_tree_list[i]) - 1
        max_depth = int(torch.max(self.class_depth).item())

        # process semantic relations
        hist_tree = torch.histc(self.class_depth,
                                bins=max_depth + 1,
                                min=0,
                                max=max_depth).int()

        index_list = []
        class_names_index_list = []
        hirarchy_level_list = []
        cls_indexes = torch.tensor(np.arange(num_classes))
        for i in range(max_depth):
            if hist_tree[i] > 1:
                hirarchy_level_list.append(i)
                index_list.append(cls_indexes[self.class_depth == i].long())
                class_names_index_list.append(class_names[index_list[-1]])
        self.hierarchy_indices_list = index_list
        self.hirarchy_level_list = hirarchy_level_list
        self.class_names_index_list = class_names_index_list

        # calcuilating normalization array
        self.normalization_factor_list = torch.zeros_like(hist_tree)
        self.normalization_factor_list[-1] = hist_tree[-1]
        for i in range(max_depth):
            self.normalization_factor_list[i] = torch.sum(hist_tree[i:], dim=0)
        self.normalization_factor_list = self.normalization_factor_list[
            0] / self.normalization_factor_list

        self.max_normalization_factor = 20
        self.normalization_factor_list.clamp_(
            max=self.max_normalization_factor)

        set_dir = os.path.join(root_dir, set_name)

        sub_class_name_list = []
        for per_sub_class_name in os.listdir(set_dir):
            sub_class_name_list.append(per_sub_class_name)
        sub_class_name_list = sorted(sub_class_name_list)

        self.image_path_list = []
        for per_sub_class_name in tqdm(sub_class_name_list):
            per_sub_class_dir = os.path.join(set_dir, per_sub_class_name)
            for per_image_name in os.listdir(per_sub_class_dir):
                if per_image_name in filter_image_name_list:
                    continue
                per_image_path = os.path.join(per_sub_class_dir,
                                              per_image_name)
                self.image_path_list.append(per_image_path)

        self.transform = transform

        print(f'Dataset Size:{len(self.image_path_list)}')
        print(f'Dataset Class Num:{len(self.class_file_name_to_single_label)}')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.load_label(idx)

        sample = {
            'image': image,
            'label': label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, idx):
        label = self.class_file_name_to_single_label[
            self.image_path_list[idx].split('/')[-2]]
        label = np.array(label)

        return label.astype(np.float32)

    def convert_outputs_to_semantic_outputs(self, outputs):
        """
        convert network outputs to 11 different hierarchies semantic outputs.
        """
        semantic_outputs = []
        for _, index in enumerate(self.hierarchy_indices_list):
            outputs_i = outputs[:, index]
            semantic_outputs.append(outputs_i)

        return semantic_outputs

    def convert_single_labels_to_semantic_labels(self, labels):
        """
        converts single labels to labels over num_of_hierarchies different hierarchies.
        [batch_size] -> [batch_size x num_of_hierarchies].
        if no hierarchical target is available, outputs -1.
        """
        device = labels.device
        numpy_labels = labels.cpu().numpy()
        batch_size = numpy_labels.shape[0]

        semantic_labels = torch.ones(
            (batch_size, len(self.hierarchy_indices_list))) * (-1.)

        for i, per_label in enumerate(numpy_labels):
            cls_multi_list = self.class_tree_list[per_label]
            hir_levels = len(cls_multi_list)
            for j, cls in enumerate(cls_multi_list):
                # protection for too small hirarchy_level_list. this protection enables us to remove hierarchies
                if len(self.hierarchy_indices_list) <= hir_levels - j - 1:
                    continue
                ind_valid = (self.hierarchy_indices_list[hir_levels - j -
                                                         1] == cls)
                semantic_labels[i,
                                hir_levels - j - 1] = torch.where(ind_valid)[0]

        semantic_labels = semantic_labels.long().to(device=device)

        return semantic_labels


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import ImageNet21K_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.classification.common import Opencv2PIL, PIL2Opencv, TorchRandomResizedCrop, TorchRandomHorizontalFlip, RandomErasing, TorchResize, TorchCenterCrop, Normalize, AutoAugment, RandAugment, ClassificationCollater

    # imagenet21ktraindataset = ImageNet21KSingleLabelDataset(
    #     root_dir=ImageNet21K_path,
    #     set_name='train',
    #     transform=transforms.Compose([
    #         Opencv2PIL(),
    #         TorchRandomResizedCrop(resize=224),
    #         TorchRandomHorizontalFlip(prob=0.5),
    #         PIL2Opencv(),
    #         # Normalize(),
    #     ]))

    # count = 0
    # for per_sample in tqdm(imagenet21ktraindataset):
    #     print(per_sample['image'].shape, per_sample['label'].shape,
    #           per_sample['label'], type(per_sample['image']),
    #           type(per_sample['label']))

    #     # temp_dir = './temp'
    #     # if not os.path.exists(temp_dir):
    #     #     os.makedirs(temp_dir)

    #     # color = [random.randint(0, 255) for _ in range(3)]
    #     # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
    #     # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     # label = per_sample['label']
    #     # text = f'label:{int(label)}'
    #     # cv2.putText(image,
    #     #             text, (30, 30),
    #     #             cv2.FONT_HERSHEY_PLAIN,
    #     #             1.5,
    #     #             color=color,
    #     #             thickness=1)

    #     # cv2.imencode('.jpg', image)[1].tofile(
    #     #     os.path.join(temp_dir, f'idx_{count}.jpg'))

    #     if count < 10:
    #         count += 1
    #     else:
    #         break

    # from torch.utils.data import DataLoader
    # collater = ClassificationCollater()
    # train_loader = DataLoader(imagenet21ktraindataset,
    #                           batch_size=128,
    #                           shuffle=True,
    #                           num_workers=2,
    #                           collate_fn=collater)

    # count = 0
    # for data in tqdm(train_loader):
    #     images, labels = data['image'], data['label']
    #     print(images.shape, labels.shape)
    #     print(images.dtype, labels.dtype)
    #     if count < 10:
    #         count += 1
    #     else:
    #         break

    # imagenet21kvaldataset = ImageNet21KSingleLabelDataset(
    #     root_dir=ImageNet21K_path,
    #     set_name='val',
    #     transform=transforms.Compose([
    #         Opencv2PIL(),
    #         TorchResize(resize=256),
    #         TorchCenterCrop(resize=224),
    #         PIL2Opencv(),
    #         Normalize(),
    #     ]))

    # count = 0
    # for per_sample in tqdm(imagenet21kvaldataset):
    #     print(per_sample['image'].shape, per_sample['label'].shape,
    #           per_sample['label'], type(per_sample['image']),
    #           type(per_sample['label']))

    #     if count < 10:
    #         count += 1
    #     else:
    #         break

    # from torch.utils.data import DataLoader
    # collater = ClassificationCollater()
    # val_loader = DataLoader(imagenet21kvaldataset,
    #                         batch_size=128,
    #                         shuffle=False,
    #                         num_workers=4,
    #                         collate_fn=collater)

    # count = 0
    # for data in tqdm(val_loader):
    #     images, labels = data['image'], data['label']
    #     print(images.shape, labels.shape)
    #     print(images.dtype, labels.dtype)
    #     if count < 10:
    #         count += 1
    #     else:
    #         break

    # imagenet21ktraindataset = ImageNet21KSingleLabelDataset(
    #     root_dir=ImageNet21K_path,
    #     set_name='train',
    #     transform=transforms.Compose([
    #         Opencv2PIL(),
    #         TorchRandomResizedCrop(resize=224),
    #         TorchRandomHorizontalFlip(prob=0.5),
    #         # AutoAugment(),
    #         RandAugment(magnitude=10, num_layers=2),
    #         PIL2Opencv(),
    #         Normalize(),
    #     ]))

    # count = 0
    # for per_sample in tqdm(imagenet21ktraindataset):
    #     print(per_sample['image'].shape, per_sample['label'].shape,
    #           per_sample['label'], type(per_sample['image']),
    #           type(per_sample['label']))

    #     if count < 10:
    #         count += 1
    #     else:
    #         break

    # from torch.utils.data import DataLoader
    # collater = ClassificationCollater()
    # train_loader = DataLoader(imagenet21ktraindataset,
    #                           batch_size=128,
    #                           shuffle=True,
    #                           num_workers=4,
    #                           collate_fn=collater)

    # count = 0
    # for data in tqdm(train_loader):
    #     images, labels = data['image'], data['label']
    #     print(images.shape, labels.shape)
    #     print(images.dtype, labels.dtype)
    #     if count < 10:
    #         count += 1
    #     else:
    #         break

    imagenet21ktraindataset = ImageNet21KSingleLabelDataset(
        root_dir=ImageNet21K_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=224),
            TorchRandomHorizontalFlip(prob=0.5),
            PIL2Opencv(),
            RandomErasing(prob=1.0, mode='pixel'),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(imagenet21ktraindataset):
        print(per_sample['image'].shape, per_sample['label'].shape,
              per_sample['label'], type(per_sample['image']),
              type(per_sample['label']))

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # color = [random.randint(0, 255) for _ in range(3)]
        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # label = per_sample['label']
        # text = f'label:{int(label)}'
        # cv2.putText(image,
        #             text, (30, 30),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             1.5,
        #             color=color,
        #             thickness=1)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 10:
            count += 1
        else:
            break

    from simpleAICV.classification.mixupcutmixclassificationcollator import MixupCutmixClassificationCollater
    from torch.utils.data import DataLoader
    collater = MixupCutmixClassificationCollater(use_mixup=True,
                                                 mixup_alpha=0.8,
                                                 cutmix_alpha=1.0,
                                                 cutmix_minmax=None,
                                                 mixup_cutmix_prob=1.0,
                                                 switch_to_cutmix_prob=0.5,
                                                 mode='batch',
                                                 correct_lam=True,
                                                 label_smoothing=0.1,
                                                 num_classes=10450)
    train_loader = DataLoader(imagenet21ktraindataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    for i, data in enumerate(tqdm(train_loader)):
        images, labels = data['image'], data['label']
        print(images.shape, labels.shape)
        print(images.dtype, labels.dtype, torch.unique(labels))

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # count = 0
        # for per_image, per_label in zip(images, labels):
        #     color = [random.randint(0, 255) for _ in range(3)]
        #     per_image = per_image.cpu().numpy()
        #     per_image = np.transpose(per_image, (1, 2, 0))
        #     image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     per_label = torch.unique(per_label)

        #     cv2.imencode('.jpg', image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{i}_count_{count}.jpg'))
        #     count += 1

        if i < 1:
            i += 1
        else:
            break

    imagenet21ktraindataset = ImageNet21KSemanticTreeLabelDataset(
        root_dir=ImageNet21K_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=224),
            TorchRandomHorizontalFlip(prob=0.5),
            PIL2Opencv(),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(imagenet21ktraindataset):
        print(per_sample['image'].shape, per_sample['label'].shape,
              per_sample['label'], type(per_sample['image']),
              type(per_sample['label']))

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # color = [random.randint(0, 255) for _ in range(3)]
        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # label = per_sample['label']
        # text = f'label:{int(label)}'
        # cv2.putText(image,
        #             text, (30, 30),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             1.5,
        #             color=color,
        #             thickness=1)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    train_loader = DataLoader(imagenet21ktraindataset,
                              batch_size=128,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, labels = data['image'], data['label']
        print(images.shape, labels.shape)
        print(images.dtype, labels.dtype)

        preds = torch.autograd.Variable(torch.randn(images.shape[0], 10450))
        semantic_preds = imagenet21ktraindataset.convert_outputs_to_semantic_outputs(
            preds)
        semantic_labels = imagenet21ktraindataset.convert_single_labels_to_semantic_labels(
            labels)
        print(preds[0], labels[0], semantic_labels[0])

        if count < 5:
            count += 1
        else:
            break
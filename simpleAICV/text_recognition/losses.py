import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CTCLoss',
    'L2Loss',
    'ACELoss',
    'AttentionLoss',
]


class CTCLoss(nn.Module):

    def __init__(self, blank_index, use_focal_weight=False, gamma=2.0):
        super(CTCLoss, self).__init__()
        self.use_focal_weight = use_focal_weight
        self.gamma = gamma
        self.loss = torch.nn.CTCLoss(blank=blank_index,
                                     reduction='none',
                                     zero_infinity=True)

    def forward(self, preds, trans_targets, input_lengths, target_lengths):
        batch = preds.shape[1]
        preds = F.log_softmax(preds, dim=2)
        loss = self.loss(preds, trans_targets, input_lengths, target_lengths)

        if self.use_focal_weight:
            pt = torch.exp(-loss)
            focal_weight = torch.pow((1. - pt), self.gamma)
            loss = focal_weight * loss

        loss = (loss / target_lengths / batch).sum()

        return loss


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, stu_preds, tea_preds):
        loss = self.loss(stu_preds, tea_preds)

        return loss


class ACELoss(nn.Module):

    def __init__(self, blank_index=0):
        super(ACELoss, self).__init__()
        self.blank_index = blank_index

    def __call__(self, preds, trans_targets):
        t, b, num_classes = preds.shape
        device = preds.device
        preds = F.softmax(preds, dim=2)
        preds = preds.mean(dim=0)

        batch_targets = torch.FloatTensor(np.zeros(
            (b, num_classes))).to(device)
        trans_targets = trans_targets.cpu().numpy()
        for idx, per_trans_target in enumerate(trans_targets):
            per_trans_target = per_trans_target[per_trans_target < num_classes]
            count_dict = collections.Counter(per_trans_target)

            for key, value in count_dict.items():
                batch_targets[idx][key] = value
            per_target_length = np.sum(per_trans_target > 0)

            batch_targets[idx][self.blank_index] = t - per_target_length

        batch_targets = batch_targets / t
        loss = (-torch.sum(torch.log(preds) * batch_targets)) / b

        return loss


class AttentionLoss(nn.Module):

    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, preds, trans_targets):
        preds = preds.permute(1, 0, 2)
        preds = preds.reshape(-1, preds.shape[-1])
        trans_targets = trans_targets.reshape(-1)
        loss = self.loss(preds, trans_targets)

        return loss


if __name__ == '__main__':
    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

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

    from simpleAICV.text_recognition.common import CTCTextLabelConverter
    from simpleAICV.text_recognition.char_sets.final_char_table import final_char_table

    preds = torch.randn(6, 127, 12113)
    input_lengths = torch.IntTensor([preds.shape[1]] * preds.shape[0])
    preds = F.log_softmax(preds, dim=2).permute(1, 0, 2)
    targets = ["你好吗", "他好吗", "大家都好吗", "nihaoma", "tahaoma", "dajiadouhaoma"]
    converter = CTCTextLabelConverter(final_char_table,
                                      str_max_length=80,
                                      garbage_char='㍿')
    trans_targets, target_lengths = converter.encode(targets)
    print("1111", preds.shape, len(input_lengths), input_lengths,
          trans_targets.shape, target_lengths)

    loss1 = CTCLoss(blank_index=0, use_focal_weight=True, gamma=2.0)
    outs = loss1(preds, trans_targets, input_lengths, target_lengths)
    print("1111", outs)

    loss2 = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    outs2 = loss2(preds, trans_targets, input_lengths, target_lengths)
    print("2222", outs2)

    stu_preds = torch.randn(6, 127, 12113)
    tea_preds = torch.randn(6, 127, 12113)
    loss3 = L2Loss()
    outs3 = loss3(stu_preds, tea_preds)
    print("3333", outs3)

    preds = torch.randn(127, 6, 12113)
    targets = ["你好吗", "他好吗", "大家都好吗", "nihaoma", "tahaoma", "dajiadouhaoma"]
    converter = CTCTextLabelConverter(final_char_table,
                                      str_max_length=80,
                                      garbage_char='㍿')
    trans_targets, target_lengths = converter.encode(targets)
    loss4 = ACELoss(blank_index=0)
    outs4 = loss4(preds, trans_targets)
    print("4444", outs4)

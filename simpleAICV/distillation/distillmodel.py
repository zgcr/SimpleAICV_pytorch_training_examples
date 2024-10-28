import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.classification import backbones
from simpleAICV.classification.common import load_state_dict

__all__ = [
    'KDModel',
]


class KDModel(nn.Module):

    def __init__(self,
                 teacher_type='resnet34',
                 student_type='resnet18',
                 teacher_pretrained_path='',
                 student_pretrained_path='',
                 freeze_teacher=True,
                 num_classes=1000,
                 use_gradient_checkpoint=False):
        super(KDModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = backbones.__dict__[teacher_type](
            **{
                'num_classes': num_classes,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })
        self.student = backbones.__dict__[student_type](
            **{
                'num_classes': num_classes,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })

        load_state_dict(teacher_pretrained_path, self.teacher)
        load_state_dict(student_pretrained_path, self.student)

        if self.freeze_teacher:
            for m in self.teacher.parameters():
                m.requires_grad = False

    def forward(self, x):
        if self.freeze_teacher:
            with torch.no_grad():
                tea_out = self.teacher(x)
        else:
            tea_out = self.teacher(x)

        stu_out = self.student(x)

        return tea_out, stu_out


if __name__ == '__main__':
    net = KDModel(teacher_type='resnet152',
                  student_type='resnet50',
                  teacher_pretrained_path='',
                  student_pretrained_path='',
                  freeze_teacher=True,
                  num_classes=1000,
                  use_gradient_checkpoint=False)
    image_h, image_w = 224, 224
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, out1_shape: {out[0].shape}, out2_shape: {out[1].shape}')

    net = KDModel(teacher_type='resnet152',
                  student_type='resnet50',
                  teacher_pretrained_path='',
                  student_pretrained_path='',
                  freeze_teacher=True,
                  num_classes=1000,
                  use_gradient_checkpoint=True)
    image_h, image_w = 224, 224
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'2222, out1_shape: {out[0].shape}, out2_shape: {out[1].shape}')

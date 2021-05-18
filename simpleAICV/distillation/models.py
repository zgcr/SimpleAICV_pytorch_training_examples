import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.classification import backbones

__all__ = [
    'KDModel',
]


class KDModel(nn.Module):
    def __init__(self,
                 teacher_type='resnet34',
                 student_type='resnet18',
                 num_classes=1000):
        super(KDModel, self).__init__()
        self.teacher = backbones.__dict__[teacher_type](
            **{
                'pretrained': True,
                'num_classes': num_classes,
            })
        self.student = backbones.__dict__[student_type](
            **{
                'pretrained': False,
                'num_classes': num_classes,
            })

        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        tea_out = self.teacher(x)
        stu_out = self.student(x)

        return tea_out, stu_out


if __name__ == '__main__':
    net = KDModel(teacher_type='resnet34',
                  student_type='resnet18',
                  num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ))
    flops, params = clever_format([flops, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, params: {params}, out1_shape: {out[0].shape}, out2_shape: {out[1].shape}'
    )

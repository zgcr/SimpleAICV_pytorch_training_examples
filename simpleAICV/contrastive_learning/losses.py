import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'DINOLoss',
]


class DINOLoss(nn.Module):

    def __init__(self,
                 planes,
                 global_crop_nums=2,
                 local_crop_nums=8,
                 warmup_teacher_temp_epochs=0,
                 teacher_temp=0.04,
                 student_temp=0.1,
                 center_momentum=0.9):
        super(DINOLoss, self).__init__()
        self.planes = planes
        self.global_crop_nums = global_crop_nums
        self.local_crop_nums = local_crop_nums
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.center = torch.zeros((1, self.planes), requires_grad=False)

    def forward(self, student_preds, teacher_preds, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student preds:[640,65536],teacher_preds:[128,65536]
        student_preds = student_preds / self.student_temp
        student_preds = student_preds.chunk(self.global_crop_nums +
                                            self.local_crop_nums)

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        per_step_teacher_temp = epoch / self.warmup_teacher_temp_epochs * self.teacher_temp if epoch < self.warmup_teacher_temp_epochs else self.teacher_temp
        self.center = self.center.to(teacher_preds.device)
        self.update_center(teacher_preds)

        # teacher centering and sharpening
        teacher_preds = F.softmax(
            (teacher_preds - self.center) / per_step_teacher_temp, dim=-1)
        teacher_preds = teacher_preds.detach().chunk(self.global_crop_nums)

        # student_preds:len(student_preds)==10,per element:[64,65536]
        # teacher_preds:len(teacher_preds)==2,per element:[64,65536]
        final_loss, loss_term_nums = 0, 0
        for teacher_idx, per_crop_teacher_pred in enumerate(teacher_preds):
            for student_idx, per_crop_student_pred in enumerate(student_preds):
                if teacher_idx == student_idx:
                    # we skip cases where student and teacher operate on the same view
                    continue
                per_crop_student_pred = F.log_softmax(per_crop_student_pred,
                                                      dim=-1)
                loss = torch.sum(-per_crop_teacher_pred *
                                 per_crop_student_pred,
                                 dim=-1)
                final_loss += loss.mean()
                loss_term_nums += 1

        final_loss /= loss_term_nums

        return final_loss

    @torch.no_grad()
    def update_center(self, teacher_preds):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_preds, dim=0, keepdim=True)
        torch.distributed.all_reduce(batch_center)
        batch_center = batch_center / (teacher_preds.shape[0] *
                                       torch.cuda.device_count())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum)
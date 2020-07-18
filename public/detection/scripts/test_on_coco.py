import time
import random
import argparse
import json
import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from public.path import COCO2017_path
from public.detection.models.retinanet import RetinaNet
from public.detection.models.fcos import FCOS
from public.detection.models.decode import RetinaDecoder, FCOSDecoder2
from public.detection.dataset.cocodataset import CocoDetection, Resize
from pycocotools.cocoeval import COCOeval


def _retinanet(arch, pretrained_model_path, num_classes):
    model = RetinaNet(arch, num_classes=num_classes)
    pretrained_models = torch.load(pretrained_model_path,
                                   map_location=torch.device('cpu'))

    # only load state_dict()
    model.load_state_dict(pretrained_models, strict=False)

    return model


def _fcos(arch, pretrained_model_path, num_classes):
    model = FCOS(arch, num_classes=num_classes)

    pretrained_models = torch.load(pretrained_model_path,
                                   map_location=torch.device('cpu'))

    # only load state_dict()
    model.load_state_dict(pretrained_models, strict=False)

    return model


def validate(val_dataset, model, decoder, args):
    if args.use_gpu:
        model = model.module
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        all_eval_result = evaluate_coco(val_dataset, model, decoder, args)

    return all_eval_result


def evaluate_coco(val_dataset, model, decoder, args):
    results, image_ids = [], []
    start_time = time.time()
    for index in range(len(val_dataset)):
        data = val_dataset[index]
        scale = data['scale']
        if args.use_gpu:
            cls_heads, reg_heads, batch_anchors = model(
                data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
        else:
            cls_heads, reg_heads, batch_anchors = model(data['img'].permute(
                2, 0, 1).float().unsqueeze(dim=0))
        scores, classes, boxes = decoder(cls_heads, reg_heads, batch_anchors)
        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        boxes /= scale

        # make sure decode batch_size=1
        # scores shape:[1,max_detection_num]
        # classes shape:[1,max_detection_num]
        # bboxes shape[1,max_detection_num,4]
        assert scores.shape[0] == 1

        scores = scores.squeeze(0)
        classes = classes.squeeze(0)
        boxes = boxes.squeeze(0)

        # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
        boxes[:, 2:] -= boxes[:, :2]

        for object_score, object_class, object_box in zip(
                scores, classes, boxes):
            object_score = float(object_score)
            object_class = int(object_class)
            object_box = object_box.tolist()
            if object_class == -1:
                break

            image_result = {
                'image_id':
                val_dataset.image_ids[index],
                'category_id':
                val_dataset.find_category_id_from_coco_label(object_class),
                'score':
                object_score,
                'bbox':
                object_box,
            }
            results.append(image_result)

        image_ids.append(val_dataset.image_ids[index])

        print('{}/{}'.format(index, len(val_dataset)), end='\r')

    testing_time = (time.time() - start_time)
    per_image_testing_time = testing_time / len(val_dataset)

    print(f"per image testing time:{per_image_testing_time:.3f}")

    if not len(results):
        print(f"No target detected in test set images")
        return

    json.dump(results,
              open('{}_bbox_results.json'.format(val_dataset.set_name), 'w'),
              indent=4)

    # load results in COCO evaluation tool
    coco_true = val_dataset.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(
        val_dataset.set_name))

    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    all_eval_result = coco_eval.stats

    return all_eval_result


def test_model(args):
    print(args)
    if args.use_gpu:
        if not torch.cuda.is_available():
            raise Exception("need gpu to test network!")
        torch.cuda.empty_cache()

    if args.seed is not None:
        random.seed(args.seed)
        if args.use_gpu:
            torch.cuda.manual_seed_all(args.seed)
            cudnn.deterministic = True

    if args.use_gpu:
        cudnn.benchmark = True
        cudnn.enabled = True

    coco_val_dataset = CocoDetection(
        image_root_dir=os.path.join(COCO2017_path, 'images/val2017'),
        annotation_root_dir=os.path.join(COCO2017_path, 'annotations'),
        set="val2017",
        transform=transforms.Compose([
            Resize(resize=args.input_image_size),
        ]))

    if args.detector == "retinanet":
        model = _retinanet(args.backbone, args.pretrained_model_path,
                           args.num_classes)
        decoder = RetinaDecoder(image_w=args.input_image_size,
                                image_h=args.input_image_size)
    elif args.detector == "fcos":
        model = _fcos(args.backbone, args.pretrained_model_path,
                      args.num_classes)
        decoder = FCOSDecoder2(image_w=args.input_image_size,
                               image_h=args.input_image_size)
    else:
        print("unsupport detection model!")
        return

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(
        f"backbone:{args.backbone},detector: '{args.detector}', flops: {flops}, params: {params}"
    )

    if args.use_gpu:
        model = model.cuda()
        decoder = decoder.cuda()
        model = nn.DataParallel(model)

    print(f"start eval.")
    all_eval_result = validate(coco_val_dataset, model, decoder, args)
    print(f"eval done.")
    if all_eval_result is not None:
        print(
            f"val: backbone: {args.backbone}, detector: {args.detector}, IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.3f}"
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch COCO Detection Distributed Training')
    parser.add_argument('--backbone', type=str, help='name of backbone')
    parser.add_argument('--detector', type=str, help='name of detector')
    parser.add_argument('--num_classes',
                        type=int,
                        default=80,
                        help='model class num')
    parser.add_argument('--pretrained_model_path',
                        type=str,
                        help='pretrained model path')
    parser.add_argument("--use_gpu",
                        action="store_true",
                        help="use gpu to test or not")
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=667,
                        help='input image size')
    args = parser.parse_args()
    test_model(args)

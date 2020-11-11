import sys
import os
import argparse
import random
import time
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from tqdm import tqdm
from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from public.path import ILSVRC2012_path
from public.imagenet.models.darknet import Darknet19, Darknet53
from public.imagenet.models.efficientnet import EfficientNet
from public.imagenet.models.regnet import RegNet
from public.imagenet.models.resnet import ResNet, BasicBlock, Bottleneck
from public.imagenet.models.vovnet import VoVNet
from public.imagenet.utils import AverageMeter, accuracy


def _darknet(arch, use_pretrained_model, pretrained_model_path, num_classes):
    if arch == 'darknet19':
        model = Darknet19(num_classes=num_classes)
    elif arch == 'darknet53':
        model = Darknet53(num_classes=num_classes)

    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def _efficientnet(arch, use_pretrained_model, pretrained_model_path,
                  num_classes):
    model = EfficientNet(arch, num_classes=num_classes)
    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def _regnet(arch, use_pretrained_model, pretrained_model_path, num_classes):
    model = RegNet(arch, num_classes=num_classes)
    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def _resnet(arch, use_pretrained_model, pretrained_model_path, num_classes,
            **kwargs):
    kwargs["num_classes"] = num_classes
    resnet_dict = {
        'resnet18': {
            "block": BasicBlock,
            "layers": [2, 2, 2, 2]
        },
        'resnet34_half': {
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
            'inplanes': 32,
        },
        'resnet34': {
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
        'resnet50_half': {
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            'inplanes': 32,
        },
        'resnet50': {
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
        'resnet101': {
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
        'resnet152': {
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    }
    resnet_config = resnet_dict[arch]

    for key, value in resnet_config.items():
        if key == "block":
            block = value
        elif key == "layers":
            layers = value
        else:
            kwargs[key] = value

    model = ResNet(block, layers, **kwargs)
    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def _vovnet(arch, use_pretrained_model, pretrained_model_path, num_classes):
    model = VoVNet(arch, num_classes=num_classes)
    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for inputs, labels in tqdm(val_loader):
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return top1.avg, top5.avg, throughput


def test_model(args):
    print(args)
    if args.use_gpu:
        # use one Graphics card to test
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    scale = 256 / 224
    val_dataset = datasets.ImageFolder(
        os.path.join(ILSVRC2012_path, 'val'),
        transforms.Compose([
            transforms.Resize(int(args.input_image_size * scale)),
            transforms.CenterCrop(args.input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    if args.classifier == "darknet":
        model = _darknet(args.backbone, args.use_pretrained_model,
                         args.pretrained_model_path, args.num_classes)
    elif args.classifier == "efficientnet":
        model = _efficientnet(args.backbone, args.use_pretrained_model,
                              args.pretrained_model_path, args.num_classes)
    elif args.classifier == "regnet":
        model = _regnet(args.backbone, args.use_pretrained_model,
                        args.pretrained_model_path, args.num_classes)
    elif args.classifier == "resnet":
        model = _resnet(args.backbone, args.use_pretrained_model,
                        args.pretrained_model_path, args.num_classes)
    elif args.classifier == "vovnet":
        model = _vovnet(args.backbone, args.use_pretrained_model,
                        args.pretrained_model_path, args.num_classes)
    else:
        print("unsupport classification model!")
        return

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(
        f"backbone:{args.backbone},classifier: '{args.classifier}', flops: {flops}, params: {params}"
    )

    if args.use_gpu:
        model = model.cuda()
        model = nn.DataParallel(model)

    print(f"start eval.")
    acc1, acc5, throughput = validate(val_loader, model, args)
    print(
        f"top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
    )
    print(f"eval done.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet Classification Testing')
    parser.add_argument('--backbone', type=str, help='name of backbone')
    parser.add_argument('--classifier', type=str, help='name of classifier')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='inference batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='num workers')
    parser.add_argument('--num_classes',
                        type=int,
                        default=1000,
                        help='model class num')
    parser.add_argument("--use_pretrained_model",
                        action="store_true",
                        help="use pretrained model or not")
    parser.add_argument('--pretrained_model_path',
                        type=str,
                        help='pretrained model path')
    parser.add_argument("--use_gpu",
                        action="store_true",
                        help="use gpu to test or not")
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=224,
                        help='input image size')
    args = parser.parse_args()
    test_model(args)

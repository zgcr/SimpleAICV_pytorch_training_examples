import numpy as np

import torch


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    only load layer parameters which has same layer name and same layer weight shape
    '''
    if not saved_model_path:
        print('No pretrained model file!')
        return

    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))

    filtered_state_dict = {
        name: weight
        for name, weight in saved_state_dict.items()
        if name in model.state_dict() and not any(
            excluded_name in name for excluded_name in excluded_layer_name)
        and weight.shape == model.state_dict()[name].shape
    }

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        model.load_state_dict(filtered_state_dict, strict=False)

    return


if __name__ == "__main__":
    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    from simpleAICV.classification import backbones

    model = backbones.__dict__['resnet50'](**{
        'num_classes': 1000,
    })
    trained_model_path = os.path.join(
        BASE_DIR,
        'classification_training/imagenet/resnet50/checkpoints/resnet50-acc76.322.pth'
    )
    load_state_dict(trained_model_path, model)
    model.eval()

    batch, channel, image_h, image_w = 1, 3, 224, 224
    images = torch.randn(batch, channel, image_h, image_w)
    x = model(images)
    print('1111', x.shape)

    pt_model = torch.jit.trace(model.cpu().eval(), (images))
    save_file_path = f'./{trained_model_path.split("/")[-1][:-4]}.pt'
    torch.jit.save(pt_model, save_file_path)

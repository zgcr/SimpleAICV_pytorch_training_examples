import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from SimpleAICV.classification.common import load_state_dict

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

    from SimpleAICV.classification import backbones

    model = backbones.__dict__['resnet50'](**{
        'num_classes': 1000,
    })
    trained_model_path = ''
    load_state_dict(trained_model_path, model)
    model.eval()

    batch, channel, image_h, image_w = 1, 3, 224, 224
    images = torch.randn(batch, channel, image_h, image_w)
    x = model(images)
    print('1111', x.shape)

    pt_model = torch.jit.trace(model.cpu().eval(), (images))
    save_file_path = f'./{trained_model_path.split("/")[-1][:-4]}.pt'
    torch.jit.save(pt_model, save_file_path)

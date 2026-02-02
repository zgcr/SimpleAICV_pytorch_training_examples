import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import numpy as np

import torch
import onnx
import onnxsim
import onnxruntime

from SimpleAICV.classification.common import load_state_dict


def convert_pytorch_model_to_onnx_model(model,
                                        inputs,
                                        save_file_path,
                                        opset_version=13,
                                        use_onnxsim=True):
    print(f'starting export with onnx version {onnx.__version__}...')
    torch.onnx.export(model,
                      inputs,
                      save_file_path,
                      export_params=True,
                      verbose=False,
                      input_names=['inputs'],
                      output_names=['outputs'],
                      opset_version=opset_version,
                      do_constant_folding=True,
                      dynamic_axes={
                          'inputs': {},
                          'outputs': {}
                      })

    # load and check onnx model
    onnx_model = onnx.load(save_file_path)
    onnx.checker.check_model(onnx_model)
    print(f'onnx model {save_file_path} is checked!')

    # Simplify onnx model
    if use_onnxsim:
        print(f'using onnx-simplifier version {onnxsim.__version__}...')
        onnx_model, check = onnxsim.simplify(
            onnx_model,
            dynamic_input_shape=False,
            input_shapes={'inputs': inputs.shape})
        assert check, 'assert onnxsim model check failed'
        onnx.save(onnx_model, save_file_path)

        print(
            f'onnxsim model is checked, convert onnxsim model success, saved as {save_file_path}'
        )


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

    save_file_path = f'./{trained_model_path.split("/")[-1][:-4]}.onnx'
    convert_pytorch_model_to_onnx_model(model, images, save_file_path)

    test_onnx_images = np.random.randn(batch, channel, image_h,
                                       image_w).astype(np.float32)
    model = onnx.load(save_file_path)
    onnxruntime_session = onnxruntime.InferenceSession(save_file_path)
    outputs = onnxruntime_session.run(None, dict(inputs=test_onnx_images))
    print('1111,onnx result:', outputs[0].shape)

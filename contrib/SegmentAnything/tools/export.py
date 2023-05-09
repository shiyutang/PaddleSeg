# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import yaml
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import paddle

from paddleseg.utils import logger, utils
from paddleseg.deploy.export import WrappedModel
from segment_anything.modeling.sam_models import SamVitB, SamVitH, SamVitL

model_link = {
    'SamVitLH':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams",
    'SamVitL':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams",
    'SamVitB':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Export Inference Model.')
    parser.add_argument("--model_type", 
        choices=['SamVitL', 'SamVitB', 'SamVitLH'], required=True,
        help="The model type.", type=str)
    parser.add_argument("--input_type", 
        choices=['boxs', 'points', 'points_grid'], required=True,
        help="The model type.", type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the exported inference model',
        type=str,
        default='./output/inference_model')
    parser.add_argument(
        "--input_img_shape",
        nargs='+',
        help="Export the model with fixed input shape, e.g., `--input_img_shape 1 3 512 1024`.",
        type=int,
        default=[1, 3, 1024, 1024])

    return parser.parse_args()


def main(args):
    utils.show_env_info()
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'

    # save model
    model = eval(args.model_type)(checkpoint=model_link[args.model_type], input_type=args.input_type)

    shape = [None, 3, None, None] if args.input_img_shape is None \
        else args.input_img_shape
    if args.input_type == 'points':
        shape2 = [1, 1, 2]
    elif args.input_type == 'boxs':
        shape2 = [1, 1, 4]
    elif args.input_type == 'points_grid':
        pass
        # todo shape2 = [None, None, 3]
    
    input_spec = [paddle.static.InputSpec(shape=shape, dtype='float32'), paddle.static.InputSpec(shape=shape2, dtype='int32'),]
    model.eval()
    model = paddle.jit.to_static(model, input_spec=input_spec)
    paddle.jit.save(model, os.path.join(args.save_dir, 'model'))

    # TODO add test config
    deploy_info = {
        'Deploy': {
            'model': 'model.pdmodel',
            'params': 'model.pdiparams',
            'input_img_shape': shape,
            'input_prompt_shape': shape2,
            'input_prompt_type': args.input_type,
            'output_dtype':  'float32'
        }
    }
    msg = '\n---------------Deploy Information---------------\n'
    msg += str(yaml.dump(deploy_info))
    logger.info(msg)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        yaml.dump(deploy_info, file)

    logger.info(f'The inference model is saved in {args.save_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

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
import cv2
import sys 
import yaml
import codecs
import argparse
import numpy as np
from typing import Any, Dict, List, Tuple
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import paddle
from paddle.nn import functional as F
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

# from paddleseg.deploy.infer import DeployConfig
from paddleseg.utils import get_image_list, logger
from paddleseg.utils.visualize import get_pseudo_color_map
from segment_anything.utils.transforms import ResizeLongestSide

class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._dir = os.path.dirname(path)

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])


def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
        and args.device == "gpu" and args.use_trt and args.enable_auto_tune


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        imgs(str, list[str], numpy): the path for images or the origin images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args), "Do not support auto_tune, which requires " \
        "device==gpu && use_trt==True && paddle >= 2.2"

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    # todo
    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        if isinstance(imgs[i], str):
            data = {'img': imgs[i]}
            data = np.array([cfg.transforms(data)['img']])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "Auto tune failed. Usually, the error is out of GPU memory "
                "for the model or image is too large. \n")
            del predictor
            if os.path.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return

    logger.info("Auto tune success.\n")



class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = DeployConfig(args.cfg)
        self.process = Processer()

        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        elif args.device == 'npu':
            self.pred_cfg.enable_custom_device('npu')
        elif args.device == 'xpu':
            self.pred_cfg.enable_xpu()
        else:
            self._init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, "
                "..., Expected all_dynamic_shape_set == true, ...', "
                "please set --enable_auto_tune=True to use auto_tune. \n")
            exit()

        if hasattr(args, 'benchmark') and args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=None,
                inference_config=self.pred_cfg,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        if not self.args.print_detail:
            self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args.precision]

        if self.args.use_trt:
            logger.info("Use TRT")
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=self.args.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                os.path.exists(self.args.auto_tuned_shape_file):
                logger.info("Use auto tuned dynamic shape")
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
                logger.info("Use manual set dynamic shape")
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 3000]}
                opt_input_shape = {"x": [1, 3, 512, 1024]}
                self.pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    def run(self, imgs_path, prompt):
        if not isinstance(imgs_path, (list, tuple)):
            imgs_path = [imgs_path]

        input_names = self.predictor.get_input_names()
        input_handle1 = self.predictor.get_input_handle(input_names[0])
        input_handle2 = self.predictor.get_input_handle(input_names[1])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        args = self.args

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        for i in range(0, len(imgs_path), args.batch_size):
            # warm up
            if i == 0 and args.benchmark:
                for j in range(5):
                    image, prompt_out = self._preprocess(imgs_path[i:i + args.batch_size][0], prompt)
                    input_handle1.reshape(image.shape)
                    input_handle1.copy_from_cpu(image)
                    input_handle2.reshape(prompt_out.shape)
                    input_handle2.copy_from_cpu(prompt_out)

                    self.predictor.run()
                    results = output_handle.copy_to_cpu()
                    results = self._postprocess(results)

            # inference
            if args.benchmark:
                self.autolog.times.start()

            image, prompt_out = self._preprocess(imgs_path[i:i + args.batch_size][0], prompt)
            input_handle1.reshape(image.shape)
            input_handle1.copy_from_cpu(image)
            input_handle2.reshape(prompt_out.shape)
            input_handle2.copy_from_cpu(prompt_out)

            if args.benchmark:
                self.autolog.times.stamp()

            self.predictor.run()

            results = output_handle.copy_to_cpu()
            if args.benchmark:
                self.autolog.times.stamp()

            results = self._postprocess(results)

            if args.benchmark:
                self.autolog.times.end(stamp=True)

            self._save_imgs(results, imgs_path[i:i + args.batch_size], args.batch_size)
        logger.info("Finish")

    def _preprocess(self, image_path, prompts):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.process.transforms(image)
        prompt = self.process.preprocess_prompt(prompts['points'], prompts['boxs'])

        return [image, prompt]

    def _postprocess(self, results):
        return self.process.postprocess(results)

    def _save_imgs(self, results, imgs_path, bs):
        for i in range(bs):
            result = get_pseudo_color_map(results[i][0])
            basename = os.path.basename(imgs_path[i])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.png'
            result.save(os.path.join(self.args.save_dir, basename))


class Processer:
    def __init__(self):
        self.image_encoder_size=1024
        self.pixel_mean: List[float]=[123.675, 116.28, 103.53]
        self.pixel_std: List[float]=[58.395, 57.12, 57.375]
        self.transform = ResizeLongestSide(self.image_encoder_size)
        self.mask_threshold = 0.0

    def transforms(self, image, image_format='RGB', ):
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)  # numpy array
        # input_image_paddle = paddle.to_tensor(input_image).cast('int32')
        # input_image_paddle = input_image_paddle.transpose(
        #     [2, 0, 1])[None, :, :, :]
        # transformed_image = input_image_paddle
        
        transformed_image =np.transpose(input_image,(2,0,1))[None,:,:,:]
        original_image_size = image.shape[:2]
        

        def preprocess(x: np.array) -> np.array:
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - np.array(self.pixel_mean)[None, :, None, None]) / np.array(self.pixel_std)[None, :, None, None]

            # Pad
            h, w = x.shape[-2:]
            padh = self.image_encoder_size - h
            padw = self.image_encoder_size - w
            x = np.pad(x, ((0,0), (0,0), (0, padh), (0, padw)))
            
            return x

        assert (
            len(transformed_image.shape) == 4 and
            transformed_image.shape[1] == 3 and
            max(*transformed_image.shape[2:]) ==
            self.image_encoder_size
        ), f"set_paddle_image input must be BCHW with long side {self.image_encoder_size}."

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = preprocess(transformed_image)

        return input_image.astype(np.float32)

    def preprocess_prompt(self, point_coords, box):
        # Transform input prompts
        if point_coords is not None:
            point_coords = self.transform.apply_coords(point_coords,
                                                    self.original_size)
            # coords_paddle = paddle.to_tensor(point_coords).cast('float32')
            coords_paddle = point_coords[None, :, :].astype('float32')
            
            return coords_paddle

        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            # box_paddle = paddle.to_tensor(box).cast('float32')
            box_paddle = box[None, :].astype('float32')
            return box_paddle
    
    def postprocess(self, low_res_masks):
        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.input_size,
                                             self.original_size)
        masks = masks > self.mask_threshold
        masks = masks.detach().cpu().numpy() # filter out the first element

        return masks
    
    def postprocess_masks(
        self,
        masks: paddle.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...], ) -> paddle.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
        masks (paddle.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
        (paddle.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = paddle.to_tensor(masks, dtype=paddle.float32)
        masks = F.interpolate(
            masks,
            (self.image_encoder_size, self.image_encoder_size),
            mode="bilinear",
            align_corners=False, )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False)
        return masks

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu', 'xpu', 'npu'],
        default="gpu",
        help="Select which device to inference, defaults to gpu.")

    parser.add_argument(
        '--use_trt',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to use Nvidia TensorRT to accelerate prediction.')
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "int8"],
        help='The tensorrt precision.')
    parser.add_argument(
        '--min_subgraph_size',
        default=3,
        type=int,
        help='The min subgraph size in tensorrt prediction.')
    parser.add_argument(
        '--enable_auto_tune',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to enable tuned dynamic shape. We uses some images to collect '
        'the dynamic shape for trt sub graph, which avoids setting dynamic shape manually.'
    )
    parser.add_argument(
        '--auto_tuned_shape_file',
        type=str,
        default="auto_tune_tmp.pbtxt",
        help='The temp file to save tuned dynamic shape.')

    parser.add_argument(
        '--cpu_threads',
        default=10,
        type=int,
        help='Number of threads to predict when using cpu.')
    parser.add_argument(
        '--enable_mkldnn',
        default=False,
        type=eval,
        choices=[True, False],
        help='Enable to use mkldnn to speed up when using cpu.')

    parser.add_argument(
        "--benchmark",
        type=eval,
        default=False,
        help="Whether to log some information about environment, model, configuration and performance."
    )
    parser.add_argument(
        "--model_name",
        default="",
        type=str,
        help='When `--benchmark` is True, the specified model name is displayed.'
    )
    parser.add_argument(
        '--point_prompt',
        type=int,
        nargs='+',
        default=None,
        help='point promt.')
    parser.add_argument(
        '--box_prompt',
        type=int,
        nargs='+',
        default=None,
        help='box promt format as xyxy.')
    parser.add_argument(
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')

    return parser.parse_args()


def main(args):
    point, box = args.point_prompt, args.box_prompt
    if point is not None:
        point = np.array([point])
        input_type = 'points'
    if box is not None:
        box = np.array([[box[0], box[1]], [box[2], box[3]]])
        input_type = 'boxs'

    imgs_list, _ = get_image_list(args.image_path)

    # collect dynamic shape by auto_tune
    if use_auto_tune(args):
        tune_img_nums = 10
        auto_tune(args, imgs_list, tune_img_nums)

    # create and run predictor
    predictor = Predictor(args)
    predictor.run(imgs_list, {'input_type': input_type, 'points': point, 'boxs':box})

    if use_auto_tune(args) and \
        os.path.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)

    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)

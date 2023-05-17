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
import time
import codecs
import argparse
import numpy as np
from typing import Any, Dict, List, Tuple
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import paddle
from paddle.vision.ops import nms
from paddle.nn import functional as F
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

# from paddleseg.deploy.infer import DeployConfig
from paddleseg.utils import get_image_list, logger
from paddleseg.utils.visualize import get_pseudo_color_map
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_paddle,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points, )


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
        self._init_params()
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

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

    def _init_params(self):
        self.points_per_side = 32
        self.points_per_batch = 64
        self.pred_iou_thresh = 0.88
        self.stability_score_thresh = 0.95
        self.stability_score_offset = 1.0
        self.box_nms_thresh = 0.7
        self.crop_n_layers = 0
        self.crop_nms_thresh = 0.7
        self.crop_overlap_ratio = 512 / 1500
        self.crop_n_points_downscale_factor = 1
        self.point_grids = None
        self.min_mask_region_area = 0
        self.output_mode = "binary_mask"
        self.model_mask_threshold = 0.0

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

            logger.info("Use manual set dynamic shape")
            min_input_shape = {"x": [1, 3, 100, 100]}
            max_input_shape = {"x": [1, 3, 2000, 3000]}
            opt_input_shape = {"x": [1, 3, 512, 1024]}
            self.pred_cfg.set_trt_dynamic_shape_info(
                min_input_shape, max_input_shape, opt_input_shape)

    def _whole_image_postprocess(self, mask_data):
        def postprocess_small_regions(mask_data: MaskData, min_area,
                                      nms_thresh) -> MaskData:
            """
            Removes small disconnected regions and holes in masks, then reruns
            box NMS to remove any new duplicates.

            Edits mask_data in place.

            Requires open-cv as a dependency.
            """
            if len(mask_data["rles"]) == 0:
                return mask_data

            # Filter small disconnected regions and holes
            new_masks = []
            scores = []
            for rle in mask_data["rles"]:
                mask = rle_to_mask(rle)

                mask, changed = remove_small_regions(
                    mask, min_area, mode="holes")
                unchanged = not changed
                mask, changed = remove_small_regions(
                    mask, min_area, mode="islands")
                unchanged = unchanged and not changed

                new_masks.append(paddle.to_tensor(mask).unsqueeze(0))
                # Give score=0 to changed masks and score=1 to unchanged masks
                # so NMS will prefer ones that didn't need postprocessing
                scores.append(float(unchanged))

            # Recalculate boxes and remove any new duplicates
            masks = paddle.concat(new_masks, axis=0)
            boxes = batched_mask_to_box(masks)
            keep_by_nms = nms(
                boxes.cast('float32'),
                scores=paddle.to_tensor(scores),
                category_idxs=paddle.zeros(len(boxes)),  # categories
                iou_threshold=nms_thresh, )

            # Only recalculate RLEs for masks that have changed
            for i_mask in keep_by_nms:
                if scores[i_mask] == 0.0:
                    mask_paddle = masks[i_mask].unsqueeze(0)
                    mask_data["rles"][i_mask] = mask_to_rle_paddle(mask_paddle)[
                        0]
                    mask_data["boxes"][i_mask] = boxes[
                        i_mask]  # update res directly
            mask_data.filter(keep_by_nms)

            return mask_data

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh), )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [
                rle_to_mask(rle) for rle in mask_data["rles"]
            ]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),  #.item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box":
                box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def run(self, image_path):

        if args.benchmark:
            for j in range(5):
                mask_data = self.run_image_no_stamp(image_path)
                results = self._whole_image_postprocess(mask_data)

        if args.benchmark:
            self.autolog.times.start()

        start = time.time()
        mask_data = self.run_image(image_path)
        results = self._whole_image_postprocess(mask_data)
        end = time.time()

        if args.benchmark:
            self.autolog.times.end(stamp=True)

        print('####inference time: %f s' % (end - start))

        self._save_imgs(results, image_path[0:0 + args.batch_size],
                        args.batch_size)
        logger.info("Finish")

    def run_image_no_stamp(self, image_path):
        image = cv2.imread(image_path[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.point_grids = build_all_layer_point_grids(
            self.points_per_side,
            self.crop_n_layers,
            self.crop_n_points_downscale_factor, )

        def box_area(boxes):
            """
            Computes the area of a set of bounding boxes, which are specified by their
            (x1, y1, x2, y2) coordinates.

            Args:
                boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                    are expected to be in (x1, y1, x2, y2) format with
                    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

            Returns:
                Tensor[N]: the area for each box
            """
            if "float" in boxes.dtype:
                boxes = boxes.cast('float64')
            elif "int" in boxes.dtype:
                boxes = boxes.cast('int64')

            boxes = boxes
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio)

        data = MaskData()

        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx,
                                           orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = nms(
                data["boxes"].cast('float32'),
                scores=scores,
                category_idxs=paddle.zeros(len(data["boxes"])),
                iou_threshold=self.crop_nms_thresh, )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def run_image(self, image_path):
        image = cv2.imread(image_path[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.point_grids = build_all_layer_point_grids(
            self.points_per_side,
            self.crop_n_layers,
            self.crop_n_points_downscale_factor, )

        def box_area(boxes):
            """
            Computes the area of a set of bounding boxes, which are specified by their
            (x1, y1, x2, y2) coordinates.

            Args:
                boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                    are expected to be in (x1, y1, x2, y2) format with
                    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

            Returns:
                Tensor[N]: the area for each box
            """
            if "float" in boxes.dtype:
                boxes = boxes.cast('float64')
            elif "int" in boxes.dtype:
                boxes = boxes.cast('int64')

            boxes = boxes
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio)

        data = MaskData()

        if args.benchmark:
            self.autolog.times.stamp()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx,
                                           orig_size)
            data.cat(crop_data)

        if args.benchmark:
            self.autolog.times.stamp()

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = nms(
                data["boxes"].cast('float32'),
                scores=scores,
                category_idxs=paddle.zeros(len(data["boxes"])),
                iou_threshold=self.crop_nms_thresh, )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
            self,
            image: np.ndarray,
            crop_box: List[int],
            crop_layer_idx: int,
            orig_size: Tuple[int, ...], ) -> MaskData:
        """
        There will be multiple forward for each cropped image here.
        """
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        # self.process.set_image(cropped_im)
        image = self.process.transforms(image)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()

        for (points, ) in batch_iterator(self.points_per_batch,
                                         points_for_image):
            batch_data = self._process_batch(image, points, cropped_im_size,
                                             crop_box, orig_size)  # forward
            data.cat(batch_data)
            del batch_data

        # Remove duplicates within this crop.
        keep_by_nms = nms(
            data["boxes"].cast('float32'),
            scores=data["iou_preds"],
            iou_threshold=self.box_nms_thresh, )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = paddle.to_tensor(
            [crop_box for _ in range(len(data["rles"]))])
        return data

    def _process_batch(
            self,
            image,
            points: np.ndarray,
            im_size: Tuple[int, ...],
            crop_box: List[int],
            orig_size: Tuple[int, ...], ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch

        in_points = self.process.preprocess_prompt(points)

        # todo properly replace and the forward, how is each image is reset? 
        masks, iou_preds = self.run_on_single_point(image,
                                                    in_points[:, None, :])

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=paddle.to_tensor(points.repeat(
                masks.shape[1], axis=0)), )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.model_mask_threshold,
            self.stability_score_offset)

        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = (
            data["masks"] > self.model_mask_threshold).cast(paddle.float32)
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box,
                                           [0, 0, orig_w, orig_h])
        if not paddle.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_paddle(data["masks"].cast(paddle.float32))
        del data["masks"]

        return data

    def run_on_single_point(self, image, point):

        input_names = self.predictor.get_input_names()
        input_handle1 = self.predictor.get_input_handle(input_names[0])
        input_handle2 = self.predictor.get_input_handle(input_names[1])

        input_handle1.reshape(image.shape)
        input_handle1.copy_from_cpu(image)
        input_handle2.reshape(point.shape)
        input_handle2.copy_from_cpu(point)

        self.predictor.run()

        output_names = self.predictor.get_output_names()
        output_handle1 = self.predictor.get_output_handle(output_names[0])
        output_handle2 = self.predictor.get_output_handle(output_names[1])

        low_res_masks = output_handle1.copy_to_cpu()
        iou_predictions = output_handle2.copy_to_cpu()

        masks = self._postprocess(low_res_masks)

        return masks, paddle.to_tensor(iou_predictions)

    def _postprocess(self, results):
        return self.process.postprocess(results)

    def _save_imgs(self, results, imgs_path, bs):
        for j in range(bs):
            result = np.ones(
                results[0]["segmentation"].shape, dtype=np.uint8) * 255
            for i, mask_data in enumerate(results):
                result[mask_data["segmentation"] == 1] = i + 1
            result = get_pseudo_color_map(result)
            basename = os.path.basename(imgs_path[j])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.png'
            result.save(os.path.join(self.args.save_dir, basename))


class Processer:
    def __init__(self):
        self.image_encoder_size = 1024
        self.pixel_mean: List[float] = [123.675, 116.28, 103.53]
        self.pixel_std: List[float] = [58.395, 57.12, 57.375]
        self.transform = ResizeLongestSide(self.image_encoder_size)
        self.mask_threshold = 0.0

    def transforms(
            self,
            image,
            image_format='RGB', ):
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)  # numpy array

        transformed_image = np.transpose(input_image, (2, 0, 1))[None, :, :, :]
        original_image_size = image.shape[:2]

        def preprocess(x: np.array) -> np.array:
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - np.array(self.pixel_mean)[None, :, None, None]
                 ) / np.array(self.pixel_std)[None, :, None, None]

            # Pad
            h, w = x.shape[-2:]
            padh = self.image_encoder_size - h
            padw = self.image_encoder_size - w
            x = np.pad(x, ((0, 0), (0, 0), (0, padh), (0, padw)))

            return x

        assert (
            len(transformed_image.shape) == 4 and
            transformed_image.shape[1] == 3 and
            max(*transformed_image.shape[2:]) == self.image_encoder_size
        ), f"set_paddle_image input must be BCHW with long side {self.image_encoder_size}."

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = preprocess(transformed_image)

        return input_image.astype(np.float32)

    def preprocess_prompt(self, point_coords):
        # Transform input prompts
        if point_coords is not None:
            point_coords = self.transform.apply_coords(point_coords,
                                                       self.original_size)
            # coords_paddle = paddle.to_tensor(point_coords).cast('float32')
            coords_paddle = point_coords.astype('float32')

            return coords_paddle

    def postprocess(self, low_res_masks):
        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.input_size,
                                       self.original_size)

        # masks = masks.detach().cpu().numpy() # filter out the first element

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
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')

    return parser.parse_args()


def infer_whole_image(args):
    imgs_list, _ = get_image_list(args.image_path)

    # create and run predictor
    predictor = Predictor(args)
    predictor.run(imgs_list)

    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parse_args()
    infer_whole_image(args)

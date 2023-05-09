# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# This implementation refers to: https://github.com/facebookresearch/segment-anything
import numpy as np
from functools import partial
from typing import Any, Dict, List, Tuple
from paddleseg.utils import load_entire_model

import paddle
from paddle import nn
from paddle.nn import functional as F

from ..modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from ..utils.transforms import ResizeLongestSide


class SAM(nn.Layer):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self,
                encoder_embed_dim,
                encoder_depth,
                encoder_num_heads,
                encoder_global_attn_indexes, checkpoint=None, input_type=None,
                pixel_mean: List[float]=[123.675, 116.28, 103.53],
                pixel_std: List[float]=[58.395, 57.12, 57.375]) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        assert input_type != None, "input_type is None, but it is required."
        self.input_type = input_type
        self.set_image = False
        
        self.image_encoder = ImageEncoderViT(
                            depth=encoder_depth,
                            embed_dim=encoder_embed_dim,
                            img_size=image_size,
                            mlp_ratio=4,
                            norm_layer=partial(
                                paddle.nn.LayerNorm, epsilon=1e-6),
                            num_heads=encoder_num_heads,
                            patch_size=vit_patch_size,
                            qkv_bias=True,
                            use_rel_pos=True,
                            global_attn_indexes=encoder_global_attn_indexes,
                            window_size=14,
                            out_chans=prompt_embed_dim, )
        self.prompt_encoder = PromptEncoder(
                            embed_dim=prompt_embed_dim,
                            image_embedding_size=(image_embedding_size, image_embedding_size),
                            input_image_size=(image_size, image_size),
                            mask_in_chans=16, )
        self.mask_decoder = MaskDecoder(
                            num_multimask_outputs=3,
                            transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=prompt_embed_dim,
                                mlp_dim=2048,
                                num_heads=8, ),
                            transformer_dim=prompt_embed_dim,
                            iou_head_depth=3,
                            iou_head_hidden_dim=256, )
        self.eval()
        if checkpoint is not None:
            load_entire_model(self, checkpoint)
        self.register_buffer(
            "pixel_mean",
            paddle.to_tensor(pixel_mean).reshape([-1, 1, 1]),
            persistable=False)
        self.register_buffer(
            "pixel_std",
            paddle.to_tensor(pixel_std).reshape([-1, 1, 1]),
            persistable=False)

    @property
    def device(self) -> Any:
        if paddle.is_compiled_with_cuda():
            return 'gpu'
        else:
            return 'cpu'
    
    def transforms(self, image, image_format='RGB', ):
        self.transform = ResizeLongestSide(self.image_encoder.img_size)

        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)  # numpy array
        input_image_paddle = paddle.to_tensor(input_image).cast('int32')

        input_image_paddle = input_image_paddle.transpose(
            [2, 0, 1])[None, :, :, :]
        
        transformed_image = input_image_paddle
        original_image_size = image.shape[:2]
        

        def preprocess(x: paddle.Tensor) -> paddle.Tensor:
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.image_encoder.img_size - h
            padw = self.image_encoder.img_size - w
            x = F.pad(x, (0, padw, 0, padh))
            return x

        assert (
            len(transformed_image.shape) == 4 and
            transformed_image.shape[1] == 3 and
            max(*transformed_image.shape[2:]) ==
            self.image_encoder.img_size
        ), f"set_paddle_image input must be BCHW with long side {self.image_encoder.img_size}."

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = preprocess(transformed_image)

        return input_image

    def preprocess_prompt(self, point_coords, point_labels, box):
        # Transform input prompts
        coords_paddle, labels_paddle, box_paddle, mask_input_paddle = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords,
                                                       self.original_size)
            coords_paddle = paddle.to_tensor(point_coords).cast('float32')
            coords_paddle = coords_paddle[None, :, :]
            
            return coords_paddle

        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_paddle = paddle.to_tensor(box).cast('float32')
            box_paddle = box_paddle[None, :]
            return box_paddle
        
    def after_forward(self):
        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()

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
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False, )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def postprocess(self, low_res_masks):
        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.input_size,
                                             self.original_size)
        masks = masks > self.mask_threshold

        return masks

    def prompt_forward_point(self, x, coords_paddle):
        labels_paddle = np.array([1])
        labels_paddle = paddle.to_tensor(labels_paddle).cast('int32')
        labels_paddle = labels_paddle[None, :]
        points = (coords_paddle, labels_paddle)
        if self.set_image == False:
            self.features = self.image_encoder(x) # [1, 3, 1024, 1024]
            # print("image_encoder shape", self.features.shape) # [1, 256, 64, 64]
            self.set_image = True
        
        # Embed prompts

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None, )
        
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True, )

        return low_res_masks #, iou_predictions, low_res_masks

    def prompt_forward_box(self, x, box_paddle):

        if self.set_image == False:
            self.features = self.image_encoder(x)
            self.set_image = True     
             
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_paddle,
            masks=None, )
        
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True)

        return low_res_masks #, iou_predictions, low_res_masks   

    def full_mask_forward(self, img: List[Dict[str, Any]], coords_paddle) :
        labels_paddle = np.array([1])
        points = (coords_paddle, labels_paddle)
        if self.set_image == False:
            self.features = self.image_encoder(img)
            self.set_image = True
        
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None, )
        
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True, )

        return low_res_masks, iou_predictions # low_res_masks,

    def forward(self, img, prompt):
        if self.input_type=='points':
            masks = self.prompt_forward_point(img, prompt)
        elif self.input_type=='boxs':
            masks = self.prompt_forward_box(img, prompt)
        elif self.input_type=='points_grid':
            masks, iou_predictions  = self.full_mask_forward(img, prompt)
        else:
            NotImplementedError('input_type need to be in {"points", "boxs", "points_grid"}, but got: {}'.format(self.input_type))

        return masks

class SamVitH(SAM):
    def __init__(self, encoder_embed_dim=1280,
                encoder_depth=32,
                encoder_num_heads=16,
                encoder_global_attn_indexes=[7, 15, 23, 31],
                checkpoint=None,
                input_type=None):
        super().__init__(encoder_embed_dim,
                        encoder_depth,
                        encoder_num_heads,
                        encoder_global_attn_indexes,checkpoint, input_type)


class SamVitL(SAM):
    def __init__(self, encoder_embed_dim=1024,
                encoder_depth=24,
                encoder_num_heads=16,
                encoder_global_attn_indexes=[5, 11, 17, 23],
                checkpoint=None, input_type=None):
        super().__init__(encoder_embed_dim,
                        encoder_depth,
                        encoder_num_heads,
                        encoder_global_attn_indexes,checkpoint, input_type)


class SamVitB(SAM):
    def __init__(self, encoder_embed_dim=768,
                encoder_depth=12,
                encoder_num_heads=12,
                encoder_global_attn_indexes=[2, 5, 8, 11],
                checkpoint=None, input_type=None):
        super().__init__(encoder_embed_dim,
                        encoder_depth,
                        encoder_num_heads,
                        encoder_global_attn_indexes,checkpoint, input_type)

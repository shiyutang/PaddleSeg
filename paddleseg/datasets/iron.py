# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class Iron(Dataset):
    """
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
    The folder structure is as follow:

        convert_annotation
        |
        |--train
        |  |--train
        |  |--val
        |  |--test
        |
        |--val
        |  |--train
        |  |--val
        |  |--test
        |--gtFine
        |  |--train
        |  |--val
        |  |--test

    Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 2

    def __init__(self,
                 transforms,
                 dataset_root="data/Mask_Iron/mask_iron/convert_annotation",
                 mode='train',
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        train_dir = os.path.join(self.dataset_root, 'train')
        val_dir = os.path.join(self.dataset_root, 'val')
        test_dir = os.path.join(self.dataset_root, 'test')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    train_dir) or not os.path.isdir(
                        val_dir) or not os.path.isdir(test_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        if mode == "train":
            data_dir = train_dir
        elif mode == 'val':
            data_dir = val_dir
        else:
            data_dir = test_dir

        img_files = sorted(glob.glob(os.path.join(data_dir, "images", '*.jpg')))

        if mode == 'test':
            self.file_list = [[img_path, ] for img_path in img_files]
        else:
            label_files = sorted(
                glob.glob(os.path.join(data_dir, "labels", '*_rawlabel.png')))

            # # resample
            # add_list_img = []
            # add_list_label = []
            # for img in img_files:
            #     img_name = os.path.split(img)[-1]
            #     assert img_name[-4] in 
            #     if img_name[:4]=='2022':
            #         add_list_img.append(img)
            #         add_list_label.append(os.path.join(data_dir, "labels", img.replace(".jpg", '_rawlabel.png')))

            # import pdb; pdb.set_trace()
            # img_files= img_files + add_list_img*15
            # label_files = label_files + add_list_img*15

            self.file_list = [
                [img_path, label_path]
                for img_path, label_path in zip(img_files, label_files)
            ]
            for img, label in self.file_list:
                img_name = os.path.split(img)[-1]
                assert img_name[:-4] in label, print(img, label)

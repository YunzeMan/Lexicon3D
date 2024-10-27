# Copyright 2021 Zhongyang Zhang
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

import random
import os.path as op
import numpy as np
import pickle as pkl
from pathlib2 import Path
import torch
import os

import torch.utils.data as data


class SvdFeature(data.Dataset):
    def __init__(self, data_dir='dataset',
                 color_range=255,
                 train=True,
                 no_augment=True,
                 aug_prob=0.5,
                 batch_size=1):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        
        self.check_files()
        self.data_dir = data_dir
        self.count = 0

    def check_files(self):
        middir = 'train' if self.train else 'val'

        info_file = os.path.join(self.data_dir, "meta_data", "scannetv2_" + middir + ".txt")

        with open(info_file) as f:
            self.scene_list = f.read().splitlines() 



    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_name = self.scene_list[idx]

        feature_file = os.path.join(self.data_dir, "lexicon3d", "svd-full", "svd-full_features", scene_name + ".pt")
        label_file = os.path.join(self.data_dir, "process_ply", "label_20", scene_name + ".pth")

        feature = torch.load(feature_file)
        label = torch.from_numpy(torch.load(label_file))

        filtered_label = label[feature['mask_full']]

        mask_2 = (filtered_label != -1)
        filtered_label = filtered_label[mask_2]
        feature = feature['feat'][mask_2]

        self.count += 1

        return feature, filtered_label


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


class Swin3dFeature(data.Dataset):
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

        feature_file = os.path.join(self.data_dir, "lexicon3d", "swin3d_unified_pairs", "feat_xyz_overlap_transformed", scene_name + ".pt")
        pose_file = os.path.join(self.data_dir, "lexicon3d", "unified_pose", scene_name + ".pt")

        feat_xyz_overlap = torch.load(feature_file)
        feat_1 = feat_xyz_overlap['feat_1'].float()
        feat_2 = feat_xyz_overlap['feat_2'].float()
        pos_1 = torch.from_numpy(feat_xyz_overlap['pos_1']).float()
        pos_2 = feat_xyz_overlap['pos_2_transformed'].float()

        overlap_1 = feat_xyz_overlap['overlap_1']
        overlap_2 = feat_xyz_overlap['overlap_2']
        overlap = torch.cat((overlap_1, overlap_2))
        rt = torch.load(pose_file)

        return feat_1, pos_1, feat_2, pos_2, overlap, rt
        
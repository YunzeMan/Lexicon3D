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

import numpy as np
import torch
from torch import nn
from functools import partial

class LinearProbeVjepa(nn.Module):
    def __init__(self, scale,
                       in_bands_num,
                       hid=64,
                       block_num=8,
                       rdn_size=3,
                       rdb_growrate=64,
                       rdb_conv_num=8,
                       mean_sen=[1.315, 1.211, 1.948, 1.892, 3.311, 6.535, 7.634, 8.197, 8.395, 8.341, 5.89, 3.616],
                       std_sen=[5.958, 2.273, 2.299, 2.668, 2.895, 4.276, 4.978, 5.237, 5.304, 5.103, 4.298, 3.3],
                       ):
        super().__init__()
        # Set all input args as attributes
        self.__dict__.update(locals())


        self.fc = nn.Linear(1024, 20)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = 1
        

    def init_norm_func(self, ref):
        self.hsi_index = np.r_[0,4:12]
        self.rgb_index = np.r_[1:4]
        mean_hsi = np.array(self.mean_sen)[self.hsi_index]
        std_hsi  = np.array(self.std_sen)[self.hsi_index]
        mean_rgb = np.array(self.mean_sen)[self.rgb_index]
        std_rgb = np.array(self.std_sen)[self.rgb_index]

        self.sub_mean = partial(mean_shift_2d,
                                mean=torch.tensor(mean_hsi, dtype=torch.float32).type_as(ref),
                                std=torch.tensor(std_hsi, dtype=torch.float32).type_as(ref),
                                base=1,
                                add=False)
        self.add_mean = partial(mean_shift_2d, 
                                mean=torch.tensor(mean_hsi, dtype=torch.float32).type_as(ref),
                                std=torch.tensor(std_hsi, dtype=torch.float32).type_as(ref),
                                base=1,
                                add=True)
        self.sub_mean_rgb = partial(mean_shift_2d, 
                                mean=torch.tensor(mean_rgb, dtype=torch.float32).type_as(ref),
                                std=torch.tensor(std_rgb, dtype=torch.float32).type_as(ref),
                                base=1,
                                add=False)

    def forward(self, x):
        pred = self.softmax(self.fc(x.float()) / self.temperature)
        return pred

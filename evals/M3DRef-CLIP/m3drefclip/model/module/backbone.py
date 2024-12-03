import torch.nn as nn
import lightning.pytorch as pl
import MinkowskiEngine as ME
from m3drefclip.model.module.common import ResidualBlock, UBlock


class Backbone(pl.LightningModule):
    def __init__(self, input_channel, output_channel, block_channels, block_reps, sem_classes):
        super().__init__()

        # 1. U-Net
        self.unet = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=input_channel, out_channels=output_channel, kernel_size=3, dimension=3),
            UBlock([output_channel * c for c in block_channels], ME.MinkowskiBatchNorm, block_reps, ResidualBlock),
            ME.MinkowskiBatchNorm(output_channel),
            ME.MinkowskiReLU(inplace=True)
        )

        # 2.1 semantic prediction branch
        self.semantic_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, sem_classes)
        )

        # 2.2 offset prediction branch
        self.offset_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, 3)
        )

    def forward(self, voxel_features, voxel_coordinates, v2p_map):
        x = ME.SparseTensor(features=voxel_features, coordinates=voxel_coordinates, device=self.device)
        unet_out = self.unet(x)
        point_features = unet_out.features[v2p_map]
        semantic_scores = self.semantic_branch(point_features)
        point_offsets = self.offset_branch(point_features)
        return point_features, semantic_scores, point_offsets

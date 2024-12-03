import torch.nn as nn
import lightning.pytorch as pl
import MinkowskiEngine as ME
from m3drefclip.model.module.common import ResidualBlock, UBlock


class TinyUnet(pl.LightningModule):
    def __init__(self, channel):
        super().__init__()

        # 1. U-Net
        self.unet = nn.Sequential(
            UBlock([channel, 2 * channel], ME.MinkowskiBatchNorm, 2, ResidualBlock),
            ME.MinkowskiBatchNorm(channel),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, proposals_voxel_feats):
        return self.unet(proposals_voxel_feats)

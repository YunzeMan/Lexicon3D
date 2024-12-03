import torch.nn as nn
import MinkowskiEngine as ME
from collections import OrderedDict
import lightning.pytorch as pl


class ResidualBlock(pl.LightningModule):

    def __init__(self, in_channels, out_channels, dimension, norm_fn=None):
        super().__init__()
        self.downsample = None
        if norm_fn is None:
            norm_fn = ME.MinkowskiBatchNorm
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, dimension=dimension)
            )

        self.conv_branch = nn.Sequential(
            norm_fn(in_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=dimension),
            norm_fn(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=dimension)
        )

    def forward(self, x):
        identity = x
        x = self.conv_branch(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        return x


class UBlock(pl.LightningModule):

    def __init__(self, n_planes, norm_fn, block_reps, block):
        super().__init__()
        self.nPlanes = n_planes
        self.D = 3
        blocks = {'block{}'.format(i): block(n_planes[0], n_planes[0], self.D, norm_fn) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = nn.Sequential(blocks)

        if len(n_planes) > 1:
            self.conv = nn.Sequential(
                norm_fn(n_planes[0]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(n_planes[0], n_planes[1], kernel_size=2, stride=2, dimension=self.D)
            )
            self.u = UBlock(n_planes[1:], norm_fn, block_reps, block)
            self.deconv = nn.Sequential(
                norm_fn(n_planes[1]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolutionTranspose(n_planes[1], n_planes[0], kernel_size=2, stride=2, dimension=self.D)
            )
            blocks_tail = {'block{}'.format(i): block(n_planes[0] * (2 - i), n_planes[0], self.D, norm_fn) for i in
                           range(block_reps)}
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = nn.Sequential(blocks_tail)

    def forward(self, x):
        out = self.blocks(x)
        identity = out
        if len(self.nPlanes) > 1:
            out = self.conv(out)
            out = self.u(out)
            out = self.deconv(out)
            out = ME.cat(identity, out)
            out = self.blocks_tail(out)
        return out


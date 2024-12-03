import torch.nn as nn
from torchvision.transforms import Normalize
import lightning.pytorch as pl


class CLIPImageEncoder(pl.LightningModule):
    def __init__(self, clip_model, output_channel, dropout):
        super().__init__()
        self.clip_model = clip_model
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_channel, output_channel),
        )

    def forward(self, x):
        output = self.clip_model.encode_image(
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(x)
        )
        output = nn.functional.normalize(output, dim=1)
        return self.mlp(output)

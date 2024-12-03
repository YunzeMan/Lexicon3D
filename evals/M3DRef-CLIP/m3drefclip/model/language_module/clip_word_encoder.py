import torch
import torch.nn as nn
import lightning.pytorch as pl


class CLIPWordEncoder(pl.LightningModule):
    def __init__(self, clip_model, output_channel, dropout):
        super().__init__()
        self.clip_model = clip_model
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_channel, output_channel),
        )
        self.text_projection = nn.Parameter(
            torch.empty(
                size=(self.clip_model.visual.output_dim, output_channel),
                device=self.device,
                dtype=torch.float32
            )
        )
        self._weight_initialization(output_channel)

    def _weight_initialization(self, output_channel):
        nn.init.normal_(self.text_projection, std=output_channel**-0.5)

    def forward(self, data_dict, output_dict):
        clip_tokens = data_dict["clip_tokens"].flatten(start_dim=0, end_dim=1)
        word_features, sentence_features = self.clip_model.encode_text(clip_tokens)
        word_features = nn.functional.normalize(word_features, dim=2)
        output_dict["word_features"] = self.mlp(word_features)
        output_dict["sentence_features"] = sentence_features @ self.text_projection

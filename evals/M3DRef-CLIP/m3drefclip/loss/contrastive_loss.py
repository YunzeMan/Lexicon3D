import lightning.pytorch as pl
import torch.nn as nn
import torch


class SinglePairContrastiveLoss(pl.LightningModule):
    def __init__(self, temperature, split_batch):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(temperature, device=self.device, dtype=torch.float32))
        self.split_batch = split_batch

    def forward(self, aabb_features, sentence_features, gt_labels, multiple_gt=False):
        aabb_features_filtered = torch.einsum("abc,adb->adc", aabb_features, gt_labels).flatten(0, 1)

        # for multiple gt labels, it takes the average of their features as the final feature
        gt_count = torch.count_nonzero(gt_labels, dim=2).flatten(0, 1)
        gt_mask = gt_count != 0

        if not gt_mask.any():
            return 0.0

        aabb_features_filtered = aabb_features_filtered[gt_mask] / gt_count[gt_mask].unsqueeze(-1)

        # normalized features
        normalized_aabb_features = nn.functional.normalize(aabb_features_filtered, dim=1)
        normalized_sentence_features = nn.functional.normalize(sentence_features[gt_mask], dim=1)

        logit_scale = self.logit_scale.exp()
        logits_1 = logit_scale * normalized_aabb_features @ normalized_sentence_features.t()
        logits_2 = logit_scale * normalized_sentence_features @ normalized_aabb_features.t()

        labels = torch.arange(normalized_aabb_features.shape[0], device=self.device, dtype=torch.uint8)  # max 255
        loss_a = nn.functional.cross_entropy(logits_1, labels)
        loss_b = nn.functional.cross_entropy(logits_2, labels)
        return (loss_a + loss_b) / 2

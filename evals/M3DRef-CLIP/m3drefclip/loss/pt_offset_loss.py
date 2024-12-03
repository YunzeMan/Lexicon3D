import torch
import lightning.pytorch as pl
import torch.nn.functional as F


class PTOffsetLoss(pl.LightningModule):
    
    def __init__(self):
        super(PTOffsetLoss, self).__init__()

    def forward(self, pred_offsets, gt_offsets, valid_mask):
        """Point-wise offset prediction losses in norm and direction

        Args:
            pred_offsets (torch.Tensor): predicted point offsets, (B, 3), float32, cuda
            gt_offsets (torch.Tensor): GT point offsets, (B, 3), float32, cuda
            valid_mask (torch.Tensor): indicate valid points involving in loss, (B,), bool, cuda

        Returns:
            torch.Tensor: [description]
        """
        if valid_mask.count_nonzero() == 0:
            # for invalid points, don't calculate loss
            return 0, 0

        valid_pred_offsets = pred_offsets[valid_mask]
        valid_gt_offsets = gt_offsets[valid_mask]
        pt_diff = valid_pred_offsets - valid_gt_offsets  # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)

        normalized_gt_offsets = F.normalize(valid_gt_offsets, p=2, dim=1, eps=torch.finfo(valid_gt_offsets.dtype).eps)
        normalized_pt_offsets = F.normalize(valid_pred_offsets, p=2, dim=1, eps=torch.finfo(valid_gt_offsets.dtype).eps)
        direction_diff = - (normalized_gt_offsets * normalized_pt_offsets).sum(-1)  # (N)

        offset_norm_loss = torch.mean(pt_dist)
        offset_direction_loss = torch.mean(direction_diff)

        return offset_norm_loss, offset_direction_loss

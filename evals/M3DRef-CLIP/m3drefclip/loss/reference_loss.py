import lightning.pytorch as pl
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from m3drefclip.util.utils import get_batch_aabb_pair_ious


class RefBCELoss(pl.LightningModule):
    def __init__(self, iou_threshold, matching_strategy, chunk_size, max_num_proposals):
        super().__init__()
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.iou_threshold = iou_threshold
        self.matching_strategy = matching_strategy
        self.chunk_size = chunk_size
        self.max_num_proposals = max_num_proposals

    def forward(self, output_dict, pred_aabbs, pred_scores, target, gt_target_objs_mask, gt_aabb_offset):
        batch_size = pred_aabbs.shape[0]
        output_dict["gt_labels"] = torch.zeros(
            size=(batch_size, self.chunk_size, self.max_num_proposals), dtype=torch.float32, device=self.device
        )
        for batch_i in range(batch_size):
            aabb_start_idx = gt_aabb_offset[batch_i]
            aabb_end_idx = gt_aabb_offset[batch_i + 1]
            for lang_i in range(self.chunk_size):
                single_aabb_mask = gt_target_objs_mask[lang_i, aabb_start_idx:aabb_end_idx]
                if torch.count_nonzero(single_aabb_mask) == 0:
                    continue
                curr_gt_aabb = target[aabb_start_idx:aabb_end_idx][single_aabb_mask]

                iou_matrix = torch.zeros(
                    size=(curr_gt_aabb.shape[0], self.max_num_proposals), dtype=pred_aabbs.dtype, device=self.device
                )
                for i, gt_aabb in enumerate(curr_gt_aabb):
                    ious = get_batch_aabb_pair_ious(
                        pred_aabbs[batch_i], gt_aabb.tile(dims=(self.max_num_proposals, 1, 1))
                    )
                    if self.matching_strategy == "all":
                        filtered_ious_indices = torch.where(ious >= self.iou_threshold)[0]
                        if filtered_ious_indices.shape[0] == 0:
                            continue
                        output_dict["gt_labels"][batch_i, lang_i, filtered_ious_indices] = 1
                    elif self.matching_strategy == "hungarian":
                        iou_matrix[i] = ious * -1
                    else:
                        raise NotImplementedError
                if self.matching_strategy == "hungarian":
                    # TODO: implement pytorch gpu version
                    row_idx, col_idx = linear_sum_assignment(iou_matrix.cpu())
                    for index in range(len(row_idx)):
                        if (iou_matrix[row_idx[index], col_idx[index]] * -1) >= self.iou_threshold:
                            output_dict["gt_labels"][batch_i, lang_i, col_idx[index]] = 1
        # self.criterion.weight = gt_aabb_dense_mask.repeat_interleave(lang_chunk_size, dim=0)
        return self.criterion(pred_scores, output_dict["gt_labels"].flatten(start_dim=0, end_dim=1))


class RefCELoss(pl.LightningModule):
    def __init__(self, iou_threshold, chunk_size, max_num_proposals):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.iou_threshold = iou_threshold
        self.chunk_size = chunk_size
        self.max_num_proposals = max_num_proposals

    def forward(self, output_dict, pred_aabb_min_max_bounds, pred_scores, gt_aabb_min_max_bounds, gt_target_objs_mask, gt_aabb_offset):
        batch_size = pred_aabb_min_max_bounds.shape[0]
        output_dict["gt_labels"] = torch.zeros(
            size=(batch_size, self.chunk_size, self.max_num_proposals), dtype=torch.float32, device=self.device
        )
        gt_aabb_min_max_bounds_filtered = torch.empty(
            size=(batch_size, self.chunk_size, 2, 3), dtype=torch.float32, device=self.device
        )
        for batch_i in range(batch_size):
            aabb_start_idx = gt_aabb_offset[batch_i]
            aabb_end_idx = gt_aabb_offset[batch_i + 1]
            # there should be only one GT aabb
            gt_aabb_min_max_bounds_filtered[batch_i] = torch.einsum(
                "abc,da->dbc", gt_aabb_min_max_bounds[aabb_start_idx:aabb_end_idx],
                gt_target_objs_mask[:, aabb_start_idx:aabb_end_idx].float()
            )
            # for lang_i in range(self.chunk_size):
            #     single_aabb_mask = gt_target_objs_mask[lang_i, aabb_start_idx:aabb_end_idx]
            #     # there should be only one GT aabb
            #     gt_aabb_min_max_bounds_filtered[batch_i, lang_i] = gt_aabb_min_max_bounds[aabb_start_idx:aabb_end_idx][single_aabb_mask][0]

        ious = get_batch_aabb_pair_ious(
            pred_aabb_min_max_bounds.unsqueeze(1).expand(size=(-1, self.chunk_size, -1, -1, -1)).reshape(-1, 2, 3),
            gt_aabb_min_max_bounds_filtered.unsqueeze(2).expand(size=(-1, -1, self.max_num_proposals, -1, -1)).reshape(-1, 2, 3)
        ).reshape(batch_size, self.chunk_size, -1)

        iou, index = ious.max(dim=2)
        passed_threshold = torch.zeros(size=(batch_size, self.chunk_size), dtype=torch.float32, device=self.device)
        passed_threshold[iou >= self.iou_threshold] = 1
        output_dict["gt_labels"].scatter_(2, index.unsqueeze(-1), passed_threshold.unsqueeze(-1))
        return self.criterion(pred_scores, output_dict["gt_labels"].flatten(start_dim=0, end_dim=1))


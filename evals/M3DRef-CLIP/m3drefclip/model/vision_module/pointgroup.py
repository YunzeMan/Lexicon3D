import torch
import torch.nn as nn
from m3drefclip.common_ops.functions import pointgroup_ops, common_ops
from m3drefclip.model.module import TinyUnet
from m3drefclip.model.module.backbone import Backbone
import lightning.pytorch as pl
import MinkowskiEngine as ME
from m3drefclip.loss.pt_offset_loss import PTOffsetLoss


class PointGroup(pl.LightningModule):
    def __init__(self, input_channel, output_channel, max_proposals, semantic_class, use_gt):
        super().__init__()

        self.backbone = Backbone(
            input_channel=input_channel, output_channel=output_channel, block_channels=[1, 2, 3, 4, 5, 6, 7],
            block_reps=2, sem_classes=semantic_class
        )

        """
            ScoreNet Block
        """
        self.score_net = TinyUnet(output_channel)
        self.score_branch = nn.Linear(output_channel, 1)
        self.output_channel = output_channel
        self.max_proposals = max_proposals
        self.use_gt = use_gt

    def forward(self, data_dict):

        batch_size = len(data_dict["scene_id"])
        output_dict = {}

        point_features, output_dict["semantic_scores"], output_dict["point_offsets"] = self.backbone(
            data_dict["voxel_features"], data_dict["voxel_xyz"], data_dict["voxel_point_map"]
        )

        if not self.use_gt:
            # get prooposal clusters
            semantic_preds = output_dict["semantic_scores"].argmax(1).to(torch.int16)

            # set mask
            semantic_preds_mask = torch.ones_like(semantic_preds, dtype=torch.bool)
            for class_label in [0, 1]:
                semantic_preds_mask = semantic_preds_mask & (semantic_preds != class_label)
            object_idxs = torch.nonzero(semantic_preds_mask).view(-1)

            batch_idxs_ = data_dict["vert_batch_ids"][object_idxs]
            batch_offsets_ = torch.cumsum(torch.bincount(batch_idxs_ + 1), dim=0, dtype=torch.int32)
            coords_ = data_dict["point_xyz"][object_idxs]
            pt_offsets_ = output_dict["point_offsets"][object_idxs]

            semantic_preds_cpu = semantic_preds[object_idxs].cpu()

            idx_shift, start_len_shift = common_ops.ballquery_batch_p(
                coords_ + pt_offsets_, batch_idxs_, batch_offsets_, 0.03, 300
            )
            cluster_obj_idxs_shift, cluster_point_idxs_shift, proposals_offset_shift = pointgroup_ops.pg_bfs_cluster(
                semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), 50
            )

            cluster_obj_idxs_shift = cluster_obj_idxs_shift.to(self.device)
            cluster_point_idxs_shift = cluster_point_idxs_shift.to(self.device)
            proposals_offset_shift = proposals_offset_shift.to(self.device)

            cluster_point_idxs_shift = object_idxs[cluster_point_idxs_shift]

            proposals_batch_id_shift_all = data_dict["vert_batch_ids"][cluster_point_idxs_shift]

            idx, start_len = common_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, 0.03, 50)

            cluster_obj_idxs, cluster_point_idxs, proposals_offset = pointgroup_ops.pg_bfs_cluster(
                semantic_preds_cpu, idx.cpu(), start_len.cpu(), 50
            )

            cluster_obj_idxs = cluster_obj_idxs.to(self.device)
            cluster_point_idxs = cluster_point_idxs.to(self.device)

            proposals_offset = proposals_offset.to(self.device)
            cluster_point_idxs = object_idxs[cluster_point_idxs]

            proposals_batch_id_all_tmp = data_dict["vert_batch_ids"][cluster_point_idxs]

            cluster_obj_idxs_shift += (proposals_offset.size(0) - 1)
            proposals_offset_shift += proposals_offset[-1]
            # proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)

            cluster_obj_idxs = torch.cat((cluster_obj_idxs, cluster_obj_idxs_shift), dim=0)
            cluster_point_idxs = torch.cat((cluster_point_idxs, cluster_point_idxs_shift), dim=0)

            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            proposals_batch_id_all = torch.cat((proposals_batch_id_all_tmp, proposals_batch_id_shift_all[1:]))
        else:
            unique_obj_ids = torch.unique(data_dict["instance_ids"])
            unique_obj_ids = unique_obj_ids[unique_obj_ids != -1]
            proposal_point_idx_list = []
            proposal_idx_list = []
            proposals_offset = torch.empty(size=(len(unique_obj_ids) + 1, ), dtype=torch.int32, device=self.device)
            proposals_offset[0] = 0
            for i, inst_id in enumerate(unique_obj_ids):
                point_idx = torch.where(data_dict["instance_ids"] == inst_id)[0]
                proposal_point_idx_list.append(point_idx)
                proposal_idx_list.append(
                    torch.full(size=(point_idx.shape[0], ), fill_value=i, device=self.device, dtype=torch.int32)
                )
                proposals_offset[i + 1] = proposals_offset[i] + point_idx.shape[0]

            cluster_obj_idxs = torch.cat(proposal_idx_list)
            cluster_point_idxs = torch.cat(proposal_point_idx_list)
            # proposals_idx = torch.hstack(
            #     (torch.cat(proposal_idx_list).unsqueeze(-1), torch.cat(proposal_point_idx_list).unsqueeze(-1))
            # )
            proposals_batch_id_all = data_dict["vert_batch_ids"][cluster_point_idxs]

        # proposals voxelization again
        proposals_voxel_feats, proposals_p2v_map, aabb_min_max_bound = clusters_voxelization(
            cluster_obj_idxs=cluster_obj_idxs,
            cluster_point_idxs=cluster_point_idxs,
            clusters_offset=proposals_offset,
            feats=point_features,
            coords=data_dict["point_xyz"],
            scale=50,
            spatial_shape=14,
            device=self.device
        )

        # score
        score_feats = self.score_net(proposals_voxel_feats)
        pt_score_feats = score_feats.features[proposals_p2v_map]  # (sumNPoint, C)
        proposals_score_feats = common_ops.roipool(pt_score_feats, proposals_offset)  # (nProposal, C)

        if not self.use_gt:
            proposals_scores = self.score_branch(proposals_score_feats).view(-1)
        else:
            proposals_scores = torch.ones(proposals_score_feats.shape[0], dtype=torch.float32, device=self.device)
        proposals_batch_id = proposals_batch_id_all[proposals_offset[:-1].long()]
        output_dict["proposal_scores"] = (proposals_scores, cluster_point_idxs, proposals_offset)
        max_num_proposal = self.max_proposals

        total_proposals = 0
        proposals_batch_offset = torch.zeros(size=(batch_size + 1,), dtype=torch.int16, device=self.device)

        proposal_batch_idx_list = []
        for b in range(batch_size):
            proposal_batch_idx = torch.nonzero(proposals_batch_id == b).squeeze(-1)
            proposal_batch_idx_list.append(proposal_batch_idx)
            pred_num = len(proposal_batch_idx) if len(proposal_batch_idx) < max_num_proposal else max_num_proposal
            total_proposals += pred_num
            proposals_batch_offset[b + 1] = total_proposals

        proposal_features = torch.zeros(
            size=(total_proposals, self.output_channel), dtype=torch.float32, device=self.device
        )

        proposal_masks_dense = torch.zeros(
            size=(batch_size, max_num_proposal), dtype=torch.bool, device=self.device
        )

        pred_aabb_min_max_bounds = torch.zeros(size=(total_proposals, 2, 3), dtype=torch.float32, device=self.device)

        # convert to batch
        for b in range(batch_size):
            proposal_batch_idx = proposal_batch_idx_list[b]

            start_idx = proposals_batch_offset[b]
            end_idx = proposals_batch_offset[b + 1]

            pred_num = end_idx - start_idx

            rearrange_ids = torch.randperm(pred_num)

            proposal_idx_sorted = proposal_batch_idx[torch.argsort(proposals_scores[proposal_batch_idx], descending=True)][0:pred_num]

            proposal_features[start_idx:end_idx] = proposals_score_feats[proposal_idx_sorted][rearrange_ids]
            pred_aabb_min_max_bounds[start_idx:end_idx] = aabb_min_max_bound[proposal_idx_sorted][rearrange_ids]

            proposal_masks_dense[b, 0:pred_num] = True

        output_dict["aabb_features"] = proposal_features
        output_dict["pred_aabb_min_max_bounds"] = pred_aabb_min_max_bounds
        output_dict["proposal_batch_offsets"] = proposals_batch_offset
        output_dict["proposal_masks_dense"] = proposal_masks_dense
        return output_dict

    def loss(self, data_dict, output_dict):
        losses = {}

        # semantic loss
        losses["semantic_loss"] = nn.functional.cross_entropy(
            output_dict["semantic_scores"], data_dict["sem_labels"], ignore_index=-1
        )

        if self.use_gt:
            return losses

        # offset loss
        gt_offsets = data_dict["instance_centers"] - data_dict["point_xyz"]
        valid = data_dict["instance_ids"] != -1
        pt_offset_criterion = PTOffsetLoss()
        losses["offset_norm_loss"], losses["offset_dir_loss"] = pt_offset_criterion(
            output_dict["point_offsets"], gt_offsets, valid_mask=valid
        )

        # score loss
        scores, cluster_point_idxs, proposals_offset = output_dict["proposal_scores"]

        ious = common_ops.get_iou(
            cluster_point_idxs, proposals_offset,
            data_dict["instance_ids"], data_dict["instance_num_point"]
        )

        gt_scores = get_segmented_scores(ious.max(1)[0], 0.75, 0.25)
        losses["score_loss"] = nn.functional.binary_cross_entropy_with_logits(scores.view(-1), gt_scores)
        return losses


def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
    """
    Args:
        scores: (N), float, 0~1

    Returns:
        segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
    """
    fg_mask = scores > fg_thresh
    bg_mask = scores < bg_thresh
    interval_mask = (fg_mask == 0) & (bg_mask == 0)

    segmented_scores = (fg_mask > 0).float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    segmented_scores[interval_mask] = scores[interval_mask] * k + b

    return segmented_scores


def clusters_voxelization(cluster_obj_idxs, cluster_point_idxs, clusters_offset, feats, coords, scale, spatial_shape, device):
    batch_idx = cluster_obj_idxs
    c_idxs = cluster_point_idxs
    feats = feats[c_idxs]
    clusters_coords = coords[c_idxs]

    clusters_coords_mean = common_ops.sec_mean(clusters_coords, clusters_offset)  # (nCluster, 3)
    clusters_coords_mean_all = torch.index_select(clusters_coords_mean, 0, batch_idx)  # (sumNPoint, 3)
    clusters_coords -= clusters_coords_mean_all

    clusters_coords_min = common_ops.sec_min(clusters_coords, clusters_offset)
    clusters_coords_max = common_ops.sec_max(clusters_coords, clusters_offset)

    aabb_min_max_bound = torch.stack(
        tensors=(clusters_coords_min + clusters_coords_mean, clusters_coords_max + clusters_coords_mean), dim=1
    )

    # 0.01 to ensure voxel_coords < spatial_shape
    clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / spatial_shape).max(1)[0] - 0.01
    clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

    min_xyz = clusters_coords_min * clusters_scale[:, None]
    max_xyz = clusters_coords_max * clusters_scale[:, None]

    clusters_scale = torch.index_select(clusters_scale, 0, batch_idx)

    clusters_coords = clusters_coords * clusters_scale[:, None]

    range = max_xyz - min_xyz
    offset = -min_xyz + torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3, device=device)
    offset += torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3, device=device)
    offset = torch.index_select(offset, 0, batch_idx)
    clusters_coords += offset

    batched_xyz = torch.cat((cluster_obj_idxs.unsqueeze(-1), clusters_coords.int()), dim=1)

    voxel_xyz, voxel_features, _, voxel_point_map = ME.utils.sparse_quantize(
        batched_xyz, feats, return_index=True, return_inverse=True, device=device.type
    )

    clusters_voxel_feats = ME.SparseTensor(features=voxel_features, coordinates=voxel_xyz, device=device)
    return clusters_voxel_feats, voxel_point_map, aabb_min_max_bound

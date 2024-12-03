
import torch
from torch.autograd import Function
import COMMON_OPS


class BallQueryBatchP(Function):
    @staticmethod
    def forward(ctx, coords, batch_idxs, batch_offsets, radius, meanActive):
        """
        :param ctx:
        :param coords: (n, 3) float
        :param batch_idxs: (n) uint8
        :param batch_offsets: (B+1) int
        :param radius: float
        :param meanActive: int
        :return: idx (nActive), int
        :return: start_len (n, 2), int
        """

        n = coords.size(0)

        assert coords.is_contiguous() and coords.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert batch_offsets.is_contiguous() and batch_offsets.is_cuda

        while True:
            idx = torch.zeros(n * meanActive, dtype=torch.int32, device="cuda")
            start_len = torch.zeros((n, 2), dtype=torch.int32, device="cuda")
            nActive = COMMON_OPS.ballquery_batch_p(coords, batch_idxs, batch_offsets, idx, start_len, n, meanActive, radius)
            if nActive <= n * meanActive:
                break
            meanActive = int(nActive // n + 1)
        idx = idx[:nActive]

        return idx, start_len

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None


ballquery_batch_p = BallQueryBatchP.apply


class SecMean(Function):
    @staticmethod
    def forward(ctx, inp, offsets):
        """
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        """
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.zeros((nProposal, C), dtype=torch.float32, device=inp.device)

        COMMON_OPS.sec_mean(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_mean = SecMean.apply


class SecMin(Function):
    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.zeros((nProposal, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.sec_min(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_min = SecMin.apply


class SecMax(Function):
    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.zeros((nProposal, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.sec_max(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_max = SecMax.apply

class RoiPool(Function):
    @staticmethod
    def forward(ctx, feats, proposals_offset):
        '''
        :param ctx:
        :param feats: (sumNPoint, C) float
        :param proposals_offset: (nProposal + 1) int
        :return: output_feats (nProposal, C) float
        '''
        nProposal = proposals_offset.size(0) - 1
        sumNPoint, C = feats.size()

        assert feats.is_contiguous()
        assert proposals_offset.is_contiguous()

        output_feats = torch.zeros((nProposal, C), dtype=torch.float32, device="cuda")
        output_maxidx = torch.zeros((nProposal, C), dtype=torch.int32, device="cuda")

        COMMON_OPS.roipool_fp(feats, proposals_offset, output_feats, output_maxidx, nProposal, C)

        ctx.for_backwards = (output_maxidx, proposals_offset, sumNPoint)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        nProposal, C = d_output_feats.size()

        output_maxidx, proposals_offset, sumNPoint = ctx.for_backwards

        d_feats = torch.zeros((sumNPoint, C), dtype=torch.float32, device="cuda")

        COMMON_OPS.roipool_bp(d_feats, proposals_offset, output_maxidx, d_output_feats.contiguous(), nProposal, C)

        return d_feats, None


roipool = RoiPool.apply

class GetIoU(Function):
    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_ids, instance_pointnum):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_ids: (N), int16, 0~total_nInst-1, -1
        :param instance_pointnum: (total_nInst), int
        :return: proposals_iou: (nProposal, total_nInst), float
        '''
        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_ids.is_contiguous() and instance_ids.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda

        proposals_iou = torch.zeros((nProposal, nInstance), dtype=torch.float32, device="cuda")

        COMMON_OPS.get_iou(proposals_idx, proposals_offset, instance_ids, instance_pointnum, proposals_iou, nInstance,
                           nProposal)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_iou = GetIoU.apply


def crop_pcd_from_aabbs(aabb_min_max_bounds, scene_points_xyz):
    output_masks = torch.zeros(
        size=(aabb_min_max_bounds.shape[0], scene_points_xyz.shape[0]), dtype=torch.bool, device=scene_points_xyz.device
    )
    COMMON_OPS.crop_pcds_from_aabbs(
        aabb_min_max_bounds.contiguous(), scene_points_xyz.contiguous(), output_masks
    )
    return output_masks


def convert_sparse_tensor_to_dense(sparse_info, idx_offsets, max_num_aabbs):
    dense_aabb_info = torch.zeros(
        size=(idx_offsets.shape[0] - 1, max_num_aabbs) + sparse_info.shape[1:],
        dtype=sparse_info.dtype, device=sparse_info.device
    )
    for i in range(idx_offsets.shape[0] - 1):
        aabb_start_idx = idx_offsets[i]
        aabb_end_idx = idx_offsets[i + 1]
        dense_aabb_info[i][0:aabb_end_idx - aabb_start_idx] = sparse_info[aabb_start_idx:aabb_end_idx]
    return dense_aabb_info

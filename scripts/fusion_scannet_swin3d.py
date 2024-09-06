import os
import torch
import importlib
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
from fusion_util import extract_lseg_img_feature, PointCloudToImageMapper, save_fused_feature_with_locs

import sys
sys.path.append('..')
sys.path.append('Swin3D_Task/SemanticSeg')
from datasets.scannet_v2 import (
    Scannetv2,
    Scannetv2_Normal,
    Scannetv2_Point,
    Scannetv2_Normal_Point,
    Scannetv2_Normal_Point_Subsample
)
from util import config
from util.data_util import collate_fn, collate_fn_pts
import MinkowskiEngine as ME
from Swin3D.modules.swin3d_layers import knn_linear_interpolation, get_offset


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of LSeg on ScanNet.')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--scan_dir', type=str, default='dataset/ScanNet/scans', help='Where is the ScanNet dataset')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for voxelization')
    parser.add_argument('--prefix', type=str, default='swin3d', help='prefix for the output file')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def main(args):   
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    scan_dir = 'dataset/ScanNet/scans/'

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    args.n_split_points = 2000000


    ##############################
    ##### load the Swin3D model ####

    args_model = config.load_cfg_from_cfg_file('Swin3D_Task/SemanticSeg/config/scannetv2/swin3D_RGBN_L.yaml')
    model_module = importlib.import_module("model.Swin3D_RGBN")
    model = model_module.Swin3D(
        depths=args_model.depths,
        channels=args_model.channels,
        num_heads=args_model.num_heads,
        window_sizes=args_model.window_size,
        up_k=args_model.up_k,
        quant_sizes=args_model.quant_size,
        drop_path_rate=args_model.drop_path_rate,
        num_classes=args_model.classes,
        num_layers=args_model.num_layers,
        stem_transformer=args_model.stem_transformer,
        upsample=args_model.upsample,
        down_stride=args_model.get("down_stride", 2),
        knn_down=args_model.get("knn_down", True),
        signal=args_model.get("signal", True),
        in_channels=args_model.get("fea_dim", 6),
        use_offset=args_model.get("use_offset", False),
        fp16_mode=args_model.get("fp16_mode", 1),
    )
    model.backbone.load_pretrained_model('swin3d/ckpt/Swin3D_RGBN_L.pth')
    model = model.eval().cuda()
    args.evaluator = model
    device = torch.device('cpu')


    val_data = Scannetv2_Normal_Point_Subsample( # =====================> Remember to change this to Scannetv2_Normal_Point
        split='new',
        data_root=scan_dir,
        voxel_size=args_model.voxel_size,
        voxel_max=800000,
        transform=None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args_model.batch_size_val,
        shuffle=False,
        num_workers=args_model.workers,
        pin_memory=True,
        sampler=None,
        collate_fn=collate_fn_pts,
    )

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    # obtain scene names from scan_dir
    data_paths = sorted(glob(join(scan_dir, "*/*_vh_clean_2.ply")))
    scene_names = [data_path.split('/')[-2] for data_path in data_paths]
    scene_names = [x for x in scene_names if "scene0044_02" not in x and "scene0586_01" not in x] # =====================> Remember to delete this line


    data_3d_paths = sorted(glob('dataset/ScanNet/openscene/scannet_3d/*/*.pth'))
    # For val_loader, only iterate id_range[0] to id_range[1]
    for i, (coord, feat, target, offset, target_pts, inverse_map) in tqdm(enumerate(val_loader)):
        scene_id = scene_names[i]
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, scene_id)
            continue

        if exists(join(out_dir, args.prefix+'_points', scene_id + '.npy')):
            print(scene_id +'.pt' + ' already exists, skip!')
            continue

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        coord, feat, target, offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        inverse_map = inverse_map.cuda(non_blocking=True)
        target_pts = target_pts.cuda(non_blocking=True)

        assert batch.shape[0] == feat.shape[0]

        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
            target_pts = target_pts[:, 0]  # for cls
        if args_model.concat_xyz:
            feat = torch.cat([feat, coord], 1)
        with torch.no_grad():
            sp_stack, coords_sp_stack = model(feat, coord, batch)
            # coords_sp.C (int) [80000, 4]    coords_sp.F (float) [80000, 10]
            # sp.C        (int) [80000, 4]    sp.F        (float) [80000, 80]


        feats = sp_stack[4].F
        xyz = coords_sp_stack[4].F[:, 1:4].detach().contiguous()
        support_xyz = coords_sp_stack[0].F[:, 1:4].detach().contiguous()
        offset = get_offset(sp_stack[4].C[:, 0])
        support_offset = get_offset(sp_stack[0].C[:, 0])
        feats_4_interpolated = knn_linear_interpolation(xyz, support_xyz, feats, offset, support_offset, K=3) # torch.Size([80020, 640])

        feats = sp_stack[2].F
        xyz = coords_sp_stack[2].F[:, 1:4].detach().contiguous()
        support_xyz = coords_sp_stack[0].F[:, 1:4].detach().contiguous()
        offset = get_offset(sp_stack[2].C[:, 0])
        support_offset = get_offset(sp_stack[0].C[:, 0])
        feats_2_interpolated = knn_linear_interpolation(xyz, support_xyz, feats, offset, support_offset, K=3) # torch.Size([80020, 320])

        # concatenate the interpolated features 4, 2, and original 0
        feats = torch.cat([feats_4_interpolated, feats_2_interpolated, sp_stack[0].F], dim=1) # torch.Size([80020, 1040])


        # use inverse_map to get the original point cloud
        n_points = inverse_map.shape[0]
        n_points_cur = n_points
        feats = feats[inverse_map]

        point_ids_all = torch.arange(n_points_cur, device=device)

        # load 3D data (point cloud) from data_3d_paths
        # first find data_3d_paths that contains scene_id
        # then load the data
        for data_3d_path in data_3d_paths:
            if scene_id in data_3d_path:
                locs_in = torch.load(data_3d_path)[0]
                break
        
        save_fused_feature_with_locs(feats, point_ids_all, locs_in, n_points, out_dir, scene_id, args)


if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    main(args)

import os
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
# add the parent directory to the path
import sys
sys.path.append('..')

import torchvision.transforms as transforms
from PIL import Image

from fusion_util import PointCloudToImageMapper, save_fused_feature_with_locs, adjust_intrinsic, make_intrinsic
from modules.vjepa import build_model as build_jepa
from modules.videomae import build_model_large
import torch.nn.functional as F
from einops import rearrange


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of LSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='val', help='split: "train"| "val"')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--prefix', type=str, default='vjepa', help='prefix for the output file')


    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):

    # short hand
    scene_id = data_path.split('/')[-1].split('_vh')[0]

    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    evaluator = args.evaluator
    transform = args.transform


    # load 3D data (point cloud)
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    if exists(join(out_dir, args.prefix+'_points', scene_id + '.npy')):
        print(scene_id +'.pt' + ' already exists, skip!')
        return 1

    def numerical_key(file):
        return int(file.split('/')[-1].split('.')[0])

    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=numerical_key)
    
    # downsample the images
    downsampled_frames = []
    for i in range(0, len(img_dirs), 160):
        downsampled_frames.extend(img_dirs[i:i+16])
    img_dirs = downsampled_frames

    num_img = len(img_dirs)
    device = torch.device('cpu')

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)

    # Process video frames from here on...
    frame_count = args.frame_count
    # for every frame_count frames in img_dirs, process the features
    for i in range(0, len(img_dirs), frame_count):
        img_dirs_chunk = img_dirs[i:i+frame_count]
        for img_id, img_dir in enumerate(img_dirs_chunk): # make video from images, and extract features
            image = Image.open(img_dir).convert('RGB')
            image = transform(image).unsqueeze(0).unsqueeze(0) # 1, 1, 3, 240, 320
            if img_id == 0:
                video_tensor = image
            else:
                video_tensor = torch.cat((video_tensor, image), dim=1) # 1, Fn, 3, 240, 320
        if video_tensor.shape[1] < frame_count: # pad the video tensor with the last frame to make it frame_count
            video_tensor = torch.cat((video_tensor, video_tensor[:, -1:, :, :, :].repeat(1, frame_count - video_tensor.shape[1], 1, 1, 1)), dim=1)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        video_tensor = video_tensor.cuda()
        try:
            with torch.no_grad():
                feat_2ds = evaluator(video_tensor)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Skipping scene due to CUDA out of memory: {scene_id}")
                return
            else:
                raise e

        feat_2ds = rearrange(feat_2ds,'b (f h w) c -> b f c h w', b = 1, f = 8, h = args.img_dim_resized[1]//16, w = args.img_dim_resized[0]//16) # 1, 8, 1024, 15, 20
        feat_2ds = feat_2ds.permute(0, 2, 1, 3, 4) # New shape: 1, 1024, 8, h//16, w//16
        # feat_2ds = F.interpolate(feat_2ds, size=(16, args.img_dim_resized[1]//16, args.img_dim_resized[0]//16), mode='trilinear', align_corners=False) # 1, 1024, 16, h//16, w//16
        feat_2ds = F.interpolate(feat_2ds, size=(16, 240, 320), mode='trilinear', align_corners=False) # 1, 1024, 16, 240, 320
        feat_2ds = feat_2ds.permute(0, 2, 1, 3, 4).squeeze(0).to(device) # 16, 1024, h//16, w//16 or 16, 1024, 240, 320
        for img_id_local, img_dir in enumerate(tqdm(img_dirs_chunk)):
            img_id = i + img_id_local
            # load pose
            posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
            pose = np.loadtxt(posepath)
            # load depth and convert to meter
            depth = imageio.v2.imread(img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale
            # calculate the 3d-2d mapping based on the depth
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue
            mapping = torch.from_numpy(mapping).to(device)
            mask = mapping[:, 3]
            vis_id[:, img_id] = mask

            feat_2d = feat_2ds[img_id_local] # 1024, h//16, w//16
            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).to(device)

            counter[mask!=0]+= 1
            sum_features[mask!=0] += feat_2d_3d[mask!=0]
        
    counter[counter==0] = 1e-5
    feat_bank = sum_features/counter
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    save_fused_feature_with_locs(feat_bank, point_ids, locs_in, n_points, out_dir, scene_id, args)


def main(args):   
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #!### Dataset specific parameters #####
    img_dim = (320, 240)
    img_dim_resized = (320, 240)
    depth_scale = 1000.0
    fx = 577.870605
    fy = 577.870605
    mx=319.5
    my=239.5
    transforms_mean = [0.48145466, 0.4578275, 0.40821073]
    transforms_std = [0.26862954, 0.26130258, 0.27577711]
    #######################################
    visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    args.img_dim_resized = img_dim_resized

    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'scannet_3d')
    data_root_2d = join(data_dir,'scannet_2d_full')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    args.feat_dim = 1024 # 512 LSeg, 768/1024 DINOv2, 640/1280 SD2.1, 1024 V-JEPA
    args.frame_count = 16
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    if split== 'train': # for training set, export a chunk of point cloud
        args.n_split_points = 300000
    else: # for the validation set, export the entire point cloud instead of chunks
        args.n_split_points = 2000000


    ##############################
    ##### load the V-JEPA model ####

    model = build_jepa().cuda()
    args.evaluator = model

    args.transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )

    # calculate image pixel-3D points correspondances
    intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
    intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim=[640, 480], image_dim=img_dim)


    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsic,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        process_one_scene(data_paths[i], out_dir, args)

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    main(args)

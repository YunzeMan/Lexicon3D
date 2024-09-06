import os
import torch
from torch import nn
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
# add the parent directory to the path
import sys
sys.path.append('..')
import open_clip
import einops as E

import torchvision.transforms as transforms
from PIL import Image

from fusion_util import PointCloudToImageMapper, save_fused_feature_with_locs, adjust_intrinsic, make_intrinsic
from utils import center_padding, resize_pos_embed, tokens_to_output


class CLIP(nn.Module):
    def __init__(
        self,
        arch="ViT-L-14",
        checkpoint="openai",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()
        assert output in ["dense-cls", "cls", "gap", "dense"]
        self.output = output
        self.checkpoint_name = "clip_" + arch.replace("-", "").lower() + checkpoint

        # Initialize a pre-trained CLIP image encoder and freeze it.
        _clip_model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=checkpoint
        )
        _clip_model = _clip_model.eval().to(torch.float32)
        self.visual = _clip_model.visual
        del _clip_model

        # Extract some attributes from CLIP module for easy access.
        self.patch_size = self.visual.conv1.stride[0]

        # get feature dimension
        feat_dim = self.visual.transformer.width
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim
        feat_dims = [feat_dim, feat_dim, feat_dim, feat_dim]

        # get extraction targets
        n_layers = len(self.visual.transformer.resblocks)
        multilayers = [
            n_layers // 4 - 1,
            n_layers // 2 - 1,
            n_layers // 4 * 3 - 1,
            n_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dims
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # clip stuff
        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")

        # concat cls token
        _cls_embed = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([_cls_embed.to(x.dtype), x], dim=1)

        # add pos embed
        pos_embed = resize_pos_embed(self.visual.positional_embedding, x_hw)
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))

        embeds = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, _x in enumerate(embeds):
            _x = tokens_to_output(self.output, _x[:, 1:], _x[:, 0], out_hw)
            outputs.append(_x)

        return outputs[0] if len(outputs) == 1 else outputs


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of CLIP on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='val', help='split: "train"| "val"')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--prefix', type=str, default='clip', help='prefix for the output file')


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

    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    device = torch.device('cpu')

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, img_dir in enumerate(tqdm(img_dirs)):

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

        image = Image.open(img_dir).convert('RGB')
        image = transform(image).unsqueeze(0).to('cuda')
        with torch.no_grad():
            feat_2d = evaluator(image)

        # feat_2d is a list with 4 elements, concatenate last 3 elements
        feat_2d = torch.cat(feat_2d[-3:], dim=1)
        # resize the feat_2d from 17x23 to 240x320
        feat_2d = torch.nn.functional.interpolate(feat_2d, size=(240, 320), mode='bicubic', align_corners=False).squeeze(0) # 240, 320, 1024
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
    img_dim_resized = (322, 238)
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

    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'scannet_3d')
    data_root_2d = join(data_dir,'scannet_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    args.feat_dim = 1024 * 3 # ========= 1024 or 1024*3
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    if split== 'train': # for training set, export a chunk of point cloud
        args.n_split_points = 300000
    else: # for the validation set, export the entire point cloud instead of chunks
        args.n_split_points = 2000000


    ##############################
    ##### load the CLIP model ####

    model = CLIP(arch="ViT-L-14", checkpoint="openai", output="dense", layer=-1, return_multilayer=True).cuda() # 238, 322 --> 17, 23
    args.evaluator = model


    args.transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=transforms_mean, std=transforms_std),
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

import os
import sys
import time
import numpy as np
import torch
import open3d as o3d
from open3d.visualization import rendering
import imageio
import matplotlib.pyplot as plt
import sklearn
import sklearn.cluster


def visualize_pc(render, pcd, pc_feat, out_dir_root, out_dir_name, scene_id):
    out_dir = os.path.join(out_dir_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # visualize pcd with color using open3d
    mtl_points = o3d.visualization.rendering.MaterialRecord()
    mtl_points.shader = "defaultUnlit"
    mtl_points.point_size = 4

    render.scene.add_geometry('point cloud', pcd, mtl_points)
    cam_dist = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()) * 0.8

    if scene_id in ['scene0031_00', 'scene0031_01']:
        pitch, yaw = 45, 180
    else:
        pitch, yaw = 45, 0

    cam_x = cam_dist * np.cos(np.deg2rad(pitch)) * np.cos(np.deg2rad(yaw))
    cam_y = cam_dist * np.cos(np.deg2rad(pitch)) * np.sin(np.deg2rad(yaw))
    cam_z = cam_dist * np.sin(np.deg2rad(pitch))
    render.setup_camera(60.0, pcd.get_center(), pcd.get_center()+[cam_x, cam_y, cam_z], [0, 0, 1])

    # render the image and save to out_dir
    img = render.render_to_image()
    img = np.array(img)
    img = img[:, :, :3]
    img = img.astype(np.uint8)
    img_path = os.path.join(out_dir, scene_id + '.png')
    imageio.imwrite(img_path, img)
    print(f"Saved {img_path}")
    # remove the point cloud from the scene
    render.scene.remove_geometry('point cloud')


def visualize_pca(render, pcd, pc_feat, out_dir_root, out_dir_name, scene_id):
    out_dir = os.path.join(out_dir_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # visualize pcd with color using open3d
    mtl_points = o3d.visualization.rendering.MaterialRecord()
    mtl_points.shader = "defaultUnlit"
    mtl_points.point_size = 4

    if scene_id in ['scene0031_00', 'scene0031_01']:
        pitch, yaw = 45, 180
    else:
        pitch, yaw = 45, 0

    cam_dist = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()) * 0.8
    cam_x = cam_dist * np.cos(np.deg2rad(pitch)) * np.cos(np.deg2rad(yaw))
    cam_y = cam_dist * np.cos(np.deg2rad(pitch)) * np.sin(np.deg2rad(yaw))
    cam_z = cam_dist * np.sin(np.deg2rad(pitch))

    # Visualize all 6 combinations of orderings of the pcd.colors (RGB, RBG, GRB, GBR, BRG, BGR)
    for i, order in enumerate([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]):
        pcd.colors = o3d.utility.Vector3dVector(pc_feat[:, order])
        render.scene.add_geometry('point cloud', pcd, mtl_points)
        render.setup_camera(60.0, pcd.get_center(), pcd.get_center()+[cam_x, cam_y, cam_z], [0, 0, 1])

        # render the image and save to out_dir
        img = render.render_to_image()
        img = np.array(img)
        img = img[:, :, :3]
        img = img.astype(np.uint8)
        img_path = os.path.join(out_dir, scene_id + f'_order_{i}.png')
        imageio.imwrite(img_path, img)
        print(f"Saved {img_path}")
        # remove the point cloud from the scene
        render.scene.remove_geometry('point cloud')


def main():
    w, h = 512, 512
    render = rendering.OffscreenRenderer(w, h)
    render.scene.view.set_post_processing(False)
    render.scene.show_axes(False)
    render.scene.scene.set_sun_light([-1, -1, -1], [2.0, 2.0, 2.0], 100000)
    render.scene.scene.enable_sun_light(True)
    render.scene.set_background([1, 1, 1, 1])

    prefix = 'clip'
    
    pc_dir =        'dataset/{prefix}/{prefix}_points'.format(prefix=prefix) 
    feat_dir =      'dataset/{prefix}/{prefix}_features'.format(prefix=prefix)
    scan_dir =      'dataset/ScanNet/scans'
    out_dir_root =  'dataset/{prefix}/{prefix}_visualization'.format(prefix=prefix)
    os.makedirs(out_dir_root, exist_ok=True)

    scene_ids = ['scene0031_00', 'scene0045_01']
    
    for scene_id in scene_ids:
        pc_pos = np.load(os.path.join(pc_dir, scene_id + '.npy'))
        pc_feat = torch.load(os.path.join(feat_dir, scene_id + '.pt'))
        pc_scan = o3d.io.read_point_cloud(os.path.join(scan_dir, scene_id, scene_id + '_vh_clean_2.ply'))

        meta_file = open(os.path.join(scan_dir, scene_id, scene_id + '.txt'), 'r').readlines()
        axis_align_matrix = None
        for line in meta_file:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
        if axis_align_matrix != None:
            axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        axis_align_matrix = axis_align_matrix if axis_align_matrix is not None else np.eye(4)
        # align the point cloud and scan with the scene
        pc_pos_4 = np.concatenate([pc_pos, np.ones((pc_pos.shape[0], 1))], axis=1)
        pc_pos_aligned = pc_pos_4 @ axis_align_matrix.transpose()
        pc_pos_aligned = pc_pos_aligned[:, :3]
        pc_scan.transform(axis_align_matrix)
        if not pc_scan.has_colors():
            print('No colors in the point cloud, skipping...')
            continue

        # set the color to (1) the strength of the sum of the features, using a colormap (like viridis in matplotlib)
        feature_strength = pc_feat['feat'].sum(dim=1)
        normalized_feature_strength = (feature_strength - feature_strength.min()) / (feature_strength.max() - feature_strength.min())
        cmap = plt.get_cmap('viridis')
        colors = cmap(normalized_feature_strength.numpy())[:, :3]  # Get RGB values, discard alpha
        pc_with_features_strength = o3d.geometry.PointCloud()
        pc_with_features_strength.points = o3d.utility.Vector3dVector(pc_pos_aligned)
        pc_with_features_strength.colors = o3d.utility.Vector3dVector(colors)
        
        # set the color to (2) the kmeans cluster of the features, using different colors for different clusters
        num_clusters = 10
        kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(pc_feat['feat'].numpy())
        clusters = kmeans.labels_ # cluster index for each point
        unique_colors = plt.get_cmap('tab10')(np.linspace(0, 1, num_clusters))[:, :3]
        # unique_colors = plt.get_cmap('rainbow')(np.linspace(0, 1, num_clusters))[:, :3]
        cluster_colors = np.array([unique_colors[cluster] for cluster in clusters])
        pc_with_feature_kmeans = o3d.geometry.PointCloud()
        pc_with_feature_kmeans.points = o3d.utility.Vector3dVector(pc_pos_aligned)
        pc_with_feature_kmeans.colors = o3d.utility.Vector3dVector(cluster_colors)

        # set the color to (3) the pca of the features
        pca = sklearn.decomposition.PCA(n_components=3)
        pca_feat = pca.fit_transform(pc_feat['feat'].numpy())
        pca_feat = (pca_feat - pca_feat.min(axis=0)) / (pca_feat.max(axis=0) - pca_feat.min(axis=0))
        pc_with_feature_pca = o3d.geometry.PointCloud()
        pc_with_feature_pca.points = o3d.utility.Vector3dVector(pc_pos_aligned)
        pc_with_feature_pca.colors = o3d.utility.Vector3dVector(pca_feat)
        
        visualize_pc(render, pc_scan, pc_feat, out_dir_root, 'rgb_scene', scene_id)
        # visualize_pc(render, pc_with_features_strength, pc_feat, out_dir_root, 'feat_scene_strength', scene_id)
        visualize_pc(render, pc_with_feature_kmeans, pc_feat, out_dir_root, 'feat_scene_kmeans', scene_id)
        visualize_pca(render, pc_with_feature_pca, pca_feat, out_dir_root, 'feat_scene_pca', scene_id)


if __name__ == '__main__':
    main()
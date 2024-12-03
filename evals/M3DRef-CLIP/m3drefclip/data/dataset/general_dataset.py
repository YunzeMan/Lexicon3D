import os
import torch
import numpy as np
from tqdm import tqdm
import random
import math
import h5py
import MinkowskiEngine as ME
from abc import abstractmethod
from torch.utils.data import Dataset


class GeneralDataset(Dataset):
    def __init__(self, data_cfg, split):
        self.data_cfg = data_cfg
        self.split = split

        # load language data from disk to memory
        self._load_language_data()

        # load scene data from disk to memory
        self._load_scene_data()

        # pack scene and language data
        self._pack_data_to_chunks()

    def _open_hdf5(self):
        self.multiview_data = h5py.File(self.data_cfg.scene_metadata.scene_multiview_file, "r", libver="latest")

    def _load_scene_data(self):
        scene_data_path = self.data_cfg.scene_dataset_path
        self.all_scene_data = {}
        for scene_id in tqdm(self.scene_ids, desc=f"Loading {self.split} data from disk"):
            scene_path = os.path.join(scene_data_path, self.split, f"{scene_id}.pth")
            scene_data = torch.load(scene_path)
            scene_data["rgb"] = scene_data["rgb"].astype(np.float32) / 127.5 - 1  # scale rgb to [-1, 1]
            self.all_scene_data[scene_id] = scene_data

    @abstractmethod
    def _load_language_data(self):
        # this function is overridden by child class
        pass

    def _pack_data_to_chunks(self):
        # this array maintains lists of pointers pointing to language and scene data
        self.chunk_lang_indices = np.empty(shape=(0, self.data_cfg.chunk_size), dtype=np.uint16)
        self.chunk_scene_indices = np.empty(shape=0, dtype=np.uint16)
        for i, scene_id in enumerate(self.scene_ids):
            num_of_chunks = math.ceil(len(self.language_data[scene_id]) / self.data_cfg.chunk_size)
            all_lang_indices = np.arange(num_of_chunks * self.data_cfg.chunk_size, dtype=np.uint16) # max 65535
            np.random.shuffle(all_lang_indices)
            chunked_lang_indices = np.split(all_lang_indices, num_of_chunks)
            chunked_scene_indices = np.full(shape=num_of_chunks, fill_value=i, dtype=np.uint16)
            self.chunk_lang_indices = np.concatenate((self.chunk_lang_indices, chunked_lang_indices), axis=0)
            self.chunk_scene_indices = np.concatenate((self.chunk_scene_indices, chunked_scene_indices), axis=0)

    def _get_xyz_augment_matrix(self):
        aug_settings = self.data_cfg.scene_augmentation
        m = np.eye(3, dtype=np.float32)
        if self.split == "train" and aug_settings.jitter_xyz:
            m += np.random.randn(3, 3) * 0.1
        if self.split == "train" and aug_settings.flip_x and random.random() > 0.5:
            m[0][0] *= -1
        if self.split == "train" and aug_settings.rotate_z:
            rad = random.choice((0, 0.5, 1, 1.5)) * np.pi  # randomly rotate around z-axis by 0, 90, 180, 270 degrees
            c = np.cos(rad)
            s = np.sin(rad)
            m = m @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return m.astype(np.float32)

    @abstractmethod
    def _augment_language(self):
        # this function is overridden by child class
        pass

    def shuffle_chunks(self):
        # called after each epoch
        self._pack_data_to_chunks()

    def __len__(self):
        return len(self.chunk_lang_indices)

    def __getitem__(self, index):
        data_dict = {}
        scene_id = self.scene_ids[self.chunk_scene_indices[index]]
        scene_data = self.all_scene_data[scene_id]
        scene_center_xyz = scene_data["xyz"].mean(axis=0)

        original_num_points = scene_data["xyz"].shape[0]
        choices = np.ones(shape=original_num_points, dtype=bool)

        # sample points
        if self.split == "train" and original_num_points > self.data_cfg.max_num_point:
            choices = np.random.choice(original_num_points, self.data_cfg.max_num_point, replace=False)

        # augment the whole scene (only applicable for the train set)
        xyz_augment_matrix = self._get_xyz_augment_matrix()
        data_dict["point_xyz"] = (scene_data["xyz"] - scene_center_xyz)[choices] @ xyz_augment_matrix
        point_normal = scene_data["normal"][choices] @ np.linalg.inv(xyz_augment_matrix).transpose()
        point_rgb = scene_data["rgb"][choices]
        data_dict["instance_ids"] = scene_data["instance_ids"][choices]
        data_dict["sem_labels"] = scene_data["sem_labels"][choices].astype(np.int64)

        data_dict["scene_center_xyz"] = scene_center_xyz  # used to recover the original pointcloud coordinates

        instance_num_point = []  # (nInst), int
        unique_instance_ids = np.unique(data_dict["instance_ids"])
        unique_instance_ids = unique_instance_ids[unique_instance_ids != -1]
        data_dict["num_instances"] = unique_instance_ids.shape[0]
        instance_centers = np.empty(shape=(data_dict["point_xyz"].shape[0], 3), dtype=np.float32)

        for index, i in enumerate(unique_instance_ids):
            assert index == i  # make sure it is consecutive
            inst_i_idx = np.where(data_dict["instance_ids"] == i)[0]
            mean_xyz_i = data_dict["point_xyz"][inst_i_idx].mean(0)  # instance_info
            instance_centers[inst_i_idx] = mean_xyz_i  # offset
            instance_num_point.append(inst_i_idx.size)  # instance_num_point

        data_dict["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)
        data_dict["instance_centers"] = instance_centers

        # TODO
        data_dict["all_point_xyz"] = (scene_data["xyz"] - scene_center_xyz) @ xyz_augment_matrix
        data_dict["all_point_rgb"] = (scene_data["rgb"] + 1) / 2

        # augment axis-aligned bounding boxes in the scene
        augmented_gt_aabb_corners_tmp = (scene_data["aabb_corner_xyz"] - scene_center_xyz) @ xyz_augment_matrix
        data_dict["gt_aabb_min_max_bounds"] = np.stack(
            (augmented_gt_aabb_corners_tmp.min(1), augmented_gt_aabb_corners_tmp.max(1)), axis=1
        )
        data_dict["gt_aabb_obj_ids"] = scene_data["aabb_obj_ids"]

        # quantize points to voxels
        point_features = np.empty(shape=(data_dict["point_xyz"].shape[0], 0), dtype=np.float32)
        if self.data_cfg.point_features.use_rgb:
            point_features = np.concatenate((point_features, point_rgb), axis=1)
        if self.data_cfg.point_features.use_normal:
            point_features = np.concatenate((point_features, point_normal), axis=1)
        if self.data_cfg.point_features.use_multiview:
            if not hasattr(self, 'multiview_data'):
                self._open_hdf5()
            point_features = np.concatenate((point_features, self.multiview_data[scene_id][()][choices]), axis=1)

        point_features = np.concatenate((point_features, data_dict["point_xyz"]), axis=1)

        data_dict["voxel_xyz"], data_dict["voxel_features"], _, data_dict["voxel_point_map"] = ME.utils.sparse_quantize(
            coordinates=data_dict["point_xyz"] - data_dict["point_xyz"].min(axis=0), features=point_features, return_index=True,
            return_inverse=True, quantization_size=self.data_cfg.voxel_size
        )
        data_dict["scene_id"] = scene_id
        return data_dict

import json
import clip
import torch
import numpy as np
from m3drefclip.data.dataset.general_dataset import GeneralDataset


class Multi3DRefer(GeneralDataset):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_language_data(self):
        # create a map, skip invalid labels to make the final semantic labels consecutive
        filtered_label_map = {}
        for i, valid_id in enumerate(self.data_cfg.scene_metadata.valid_semantic_mapping):
            filtered_label_map[valid_id] = i

        # load language data
        file_path = getattr(self.data_cfg.lang_metadata, f"{self.split}_language_data")
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        self.language_data = {}
        scene_ids = {}
        for item in raw_data:
            scene_ids[item["scene_id"]] = True
            if item["scene_id"] not in self.language_data:
                self.language_data[item["scene_id"]] = []

            object_name = item["object_name"].replace("_", " ")
            self.language_data[item["scene_id"]].append(
                {
                    "object_ids": np.array(item["object_ids"], dtype=np.int16),
                    "object_name": object_name,
                    "ann_id": item["ann_id"],
                    "eval_type": item["eval_type"],
                    "clip_tokens": clip.tokenize(item["description"].strip(), truncate=True)[0]
                }
            )
        self.scene_ids = list(scene_ids.keys())

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)  # get scene data from parent class

        # prepare language data
        scene_id = self.scene_ids[self.chunk_scene_indices[index]]
        language_data_indices = self.chunk_lang_indices[index]
        language_data_in_scene = self.language_data[scene_id]
        num_language_data_in_scene = len(language_data_in_scene)

        data_dict["ann_id"] = np.empty(shape=self.data_cfg.chunk_size, dtype=np.int32)
        data_dict["object_id"] = np.empty(shape=self.data_cfg.chunk_size, dtype=np.int16)

        data_dict["gt_target_obj_id_mask"] = np.zeros(
            shape=(self.data_cfg.chunk_size, data_dict["gt_aabb_obj_ids"].shape[0]), dtype=bool
        )

        data_dict["clip_tokens"] = torch.empty(size=(self.data_cfg.chunk_size, 77), dtype=torch.int32)
        data_dict["eval_type"] = []
        for i, index in enumerate(language_data_indices):
            real_idx = index % num_language_data_in_scene  # pad the last chunk
            data = language_data_in_scene[real_idx]
            data_dict["ann_id"][i] = data["ann_id"]
            data_dict["object_id"][i] = 0  # dummy value for multi3drefer dataset
            data_dict["gt_target_obj_id_mask"][i] = np.in1d(data_dict["gt_aabb_obj_ids"], data["object_ids"])
            data_dict["clip_tokens"][i] = data["clip_tokens"]
            data_dict["eval_type"].append(data["eval_type"])
        return data_dict
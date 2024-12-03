from m3drefclip.data.dataset.general_dataset import GeneralDataset
import numpy as np
import torch
import clip
import csv


class Nr3D(GeneralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_language_data(self):
        # create a map, skip invalid labels to make the final semantic labels consecutive
        filtered_label_map = {}
        for i, valid_id in enumerate(self.data_cfg.scene_metadata.valid_semantic_mapping):
            filtered_label_map[valid_id] = i

        file_path = getattr(self.data_cfg.lang_metadata, f"{self.split}_language_data")

        raw_data = []
        with open(file_path, "r") as f:
            csv_data = csv.DictReader(f)
            for row in csv_data:
                raw_data.append(row)

        # TODO
        tmp_ann_id_count = {}
        self.language_data = {}
        scene_ids = {}
        for item in raw_data:
            scene_ids[item["scan_id"]] = True
            # TODO
            scene_obj_key = (item["scan_id"], int(item["target_id"]))
            if scene_obj_key not in tmp_ann_id_count:
                tmp_ann_id_count[scene_obj_key] = 0
            else:
                tmp_ann_id_count[scene_obj_key] += 1
            if item["scan_id"] not in self.language_data:
                self.language_data[item["scan_id"]] = []

            is_easy = item["is_easy"] == "True"
            is_view_dep = item["is_view_dep"] == "True"
            if is_easy and is_view_dep:
                eval_type = "easy_dep"
            elif is_easy:
                eval_type = "easy_indep"
            elif is_view_dep:
                eval_type = "hard_dep"
            else:
                eval_type = "hard_indep"
            self.language_data[item["scan_id"]].append(
                {
                    "object_id": int(item["target_id"]),
                    "object_name": item["instance_type"],
                    "ann_id": tmp_ann_id_count[scene_obj_key],
                    "eval_type": eval_type,
                    "clip_tokens": clip.tokenize(item["utterance"].strip(), truncate=True)[0]
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
            data_dict["object_id"][i] = data["object_id"]
            data_dict["gt_target_obj_id_mask"][i] = np.in1d(data_dict["gt_aabb_obj_ids"], data["object_id"])
            data_dict["clip_tokens"][i] = data["clip_tokens"]
            data_dict["eval_type"].append(data["eval_type"])
        return data_dict

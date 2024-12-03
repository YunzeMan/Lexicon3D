"""
This help file adds evaluation labels (unique / multiple) to original ScanRefer data, it follows the
official code logic. Please refer to
https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py
"""

from tqdm.contrib.concurrent import process_map
from functools import partial
import hydra
import json
import csv


def get_semantic_mapping_file(file_path, mapping_name):
    label_mapping = {}
    mapping_col_idx = {
        "nyu40": 4,
        "eigen13": 5,
        "mpcat40": 16
    }
    with open(file_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        next(tsv_file)  # skip the header
        for line in tsv_file:
            label_mapping[line[1]] = int(line[mapping_col_idx[mapping_name]])
    return label_mapping


def add_unique_multiple_labels_to_json(file_path, label_mapping, valid_semantic_mapping):
    with open(file_path, "r") as f:
        scanrefer_json_data = json.load(f)
    obj_cache = {}
    sem_cache = {}
    for item in scanrefer_json_data:
        if (item["scene_id"], item["object_id"]) in obj_cache:
            continue
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39
        if (item['scene_id'], sem_label) not in sem_cache:
            sem_cache[(item['scene_id'], sem_label)] = 0
        sem_cache[(item['scene_id'], sem_label)] += 1
        obj_cache[(item["scene_id"], item["object_id"])] = True

    for item in scanrefer_json_data:
        scene_id = item['scene_id']
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39
        assert sem_cache[(scene_id, sem_label)] >= 1
        item["eval_type"] = "unique" if sem_cache[(scene_id, sem_label)] == 1 else "multiple"
    # save the new json
    with open(file_path, "w") as f:
        json.dump(scanrefer_json_data, f, indent=2)


@hydra.main(version_base=None, config_path="../../config", config_name="global_config")
def main(cfg):
    print("\nDefault: using all CPU cores.")
    label_mapping = get_semantic_mapping_file(cfg.data.scene_metadata.label_mapping_file, "nyu40")
    file_paths = []
    for split in ("train", "val", "test"):
        file_paths.append(getattr(cfg.data.lang_metadata, f"{split}_language_data"))

    process_map(
        partial(
            add_unique_multiple_labels_to_json, label_mapping=label_mapping,
            valid_semantic_mapping=cfg.data.scene_metadata.valid_semantic_mapping
        ), file_paths, chunksize=1
    )

    print(f"==> Complete.")


if __name__ == '__main__':
    main()

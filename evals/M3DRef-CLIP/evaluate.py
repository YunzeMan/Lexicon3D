from tqdm import tqdm
import numpy as np
import hydra
import torch
import json
import csv
import os


def generate_gt_scanrefer(split, lang_input_path, scene_root_path):
    gt_dict = {}
    scene_ids = {}
    with open(lang_input_path, "r") as f:
        raw_data = json.load(f)
    for query in tqdm(raw_data, desc="Initializing ground truths"):
        scene_id = query["scene_id"]
        scene_ids[scene_id] = True
        object_id = int(query["object_id"])
        scene_data = torch.load(os.path.join(scene_root_path, split, f"{scene_id}.pth"))
        object_ids = [object_id]
        corners = scene_data["aabb_corner_xyz"][np.in1d(scene_data["aabb_obj_ids"], np.array(object_ids))]
        aabb_min_max_bound = np.stack((corners.min(1), corners.max(1)), axis=1)
        gt_dict[(scene_id, object_id, int(query["ann_id"]))] = {
            "aabb_bound": aabb_min_max_bound,
            "eval_type": query["eval_type"]
        }
    scene_ids = list(scene_ids.keys())
    return gt_dict, scene_ids


def generate_gt_multi3drefer(split, lang_input_path, scene_root_path):
    gt_dict = {}
    scene_ids = {}
    with open(lang_input_path, "r") as f:
        raw_data = json.load(f)
    for query in tqdm(raw_data, desc="Initializing ground truths"):
        scene_id = query["scene_id"]
        scene_ids[scene_id] = True
        scene_data = torch.load(os.path.join(scene_root_path, split, f"{scene_id}.pth"))
        object_ids = query["object_ids"]
        corners = scene_data["aabb_corner_xyz"][np.in1d(scene_data["aabb_obj_ids"], np.array(object_ids))]
        aabb_min_max_bound = np.stack((corners.min(1), corners.max(1)), axis=1)
        gt_dict[(scene_id, 0, query["ann_id"])] = {
            "aabb_bound": aabb_min_max_bound,
            "eval_type": query["eval_type"]
        }
    scene_ids = list(scene_ids.keys())
    return gt_dict, scene_ids


def generate_gt_nr3d(split, lang_input_path, scene_root_path):
    gt_dict = {}
    scene_ids = {}
    tmp_ann_id_count = {}
    raw_data = []
    with open(lang_input_path, "r") as f:
        csv_data = csv.DictReader(f)
        for row in csv_data:
            raw_data.append(row)

    for query in tqdm(raw_data, desc="Initializing ground truths"):
        scene_id = query["scan_id"]
        scene_ids[scene_id] = True
        object_id = int(query["target_id"])

        scene_obj_key = (scene_id, object_id)
        if scene_obj_key not in tmp_ann_id_count:
            tmp_ann_id_count[scene_obj_key] = 0
        else:
            tmp_ann_id_count[scene_obj_key] += 1

        scene_data = torch.load(os.path.join(scene_root_path, split, f"{scene_id}.pth"))
        object_ids = [object_id]
        corners = scene_data["aabb_corner_xyz"][np.in1d(scene_data["aabb_obj_ids"], np.array(object_ids))]
        aabb_min_max_bound = np.stack((corners.min(1), corners.max(1)), axis=1)

        is_easy = query["is_easy"] == "True"
        is_view_dep = query["is_view_dep"] == "True"
        if is_easy and is_view_dep:
            eval_type = "easy_dep"
        elif is_easy:
            eval_type = "easy_indep"
        elif is_view_dep:
            eval_type = "hard_dep"
        else:
            eval_type = "hard_indep"

        gt_dict[(scene_id, object_id, tmp_ann_id_count[scene_obj_key])] = {
            "aabb_bound": aabb_min_max_bound,
            "eval_type": eval_type
        }
    scene_ids = list(scene_ids.keys())
    return gt_dict, scene_ids


def parse_prediction(scene_ids, pred_file_root_path):
    pred_dict = {}
    for scene_id in scene_ids:
        with open(os.path.join(pred_file_root_path, f"{scene_id}.json"), "r") as f:
            scene_predictions = json.load(f)
        for scene_prediction in scene_predictions:
            corners = np.array(scene_prediction["aabb"], dtype=np.float32)
            if corners.shape[0] == 0:
                corners = np.empty(shape=(0, 8, 3), dtype=np.float32)
            aabb_min_max_bound = np.stack((corners.min(1), corners.max(1)), axis=1)
            if "object_id" in scene_prediction:
                object_id = int(scene_prediction["object_id"])
            else:
                # for Multi3DRefer, set a dummy object_id
                object_id = 0
            pred_dict[(scene_id, object_id, int(scene_prediction["ann_id"]))] = {
                "aabb_bound": aabb_min_max_bound
            }
    return pred_dict


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    split = cfg.data.evaluation.split

    # prepare gt
    lang_input_path = getattr(cfg.data.lang_metadata, f"{split}_language_data")

    assert os.path.exists(cfg.pred_path), f"Error: Predictions file path {cfg.pred_path} does not exist."

    if cfg.data.lang_dataset == "ScanRefer":
        gt_data, scene_ids = generate_gt_scanrefer(split, lang_input_path, cfg.data.scene_dataset_path)
    elif cfg.data.lang_dataset == "Nr3D":
        gt_data, scene_ids = generate_gt_nr3d(split, lang_input_path, cfg.data.scene_dataset_path)
    elif cfg.data.lang_dataset == "Multi3DRefer":
        gt_data, scene_ids = generate_gt_multi3drefer(split, lang_input_path, cfg.data.scene_dataset_path)
    else:
        raise NotImplementedError

    # prepare predictions
    pred_data = parse_prediction(scene_ids, cfg.pred_path)

    evaluator = hydra.utils.instantiate(cfg.data.evaluator, verbose=True)
    evaluator.set_ground_truths(gt_data)

    _ = evaluator.evaluate(pred_data)


if __name__ == '__main__':
    main()

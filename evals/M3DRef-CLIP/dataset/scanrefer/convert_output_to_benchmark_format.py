import hydra
import json
import os


@hydra.main(version_base=None, config_path="../../config", config_name="global_config")
def main(cfg):
    # we need to get unique/multiple labels from the dataset json
    unique_multiple_dict = {}
    with open(cfg.data.lang_metadata.test_language_data, "r") as f:
        dataset_json = json.load(f)
    for item in dataset_json:
        unique_multiple_dict[item["scene_id"], int(item["object_id"]), int(item["ann_id"])] = 1 if item["eval_type"] == "multiple" else 0

    outputs = []

    prediction_file_names = os.listdir(cfg.pred_path)
    for prediction_file_name in prediction_file_names:
        file_path = os.path.join(cfg.pred_path, prediction_file_name)
        with open(file_path, "r") as f:
            input_json = json.load(f)
        for item in input_json:
            output_item = {
                "scene_id": prediction_file_name[:-5],
                "object_id": item["object_id"],
                "ann_id": item["ann_id"],
                "bbox": item["aabb"][0],
                "unique_multiple": unique_multiple_dict[(prediction_file_name[:-5], item["object_id"], item["ann_id"])],
                "others": 0  # dummy value
            }
            outputs.append(output_item)

    os.makedirs(cfg.output_path, exist_ok=True)
    with open(os.path.join(cfg.output_path, "benchmark_output.json"), "w") as f:
        json.dump(outputs, f)

    print(f"==> Complete. Saved at: {os.path.abspath(cfg.output_path)}")


if __name__ == '__main__':
    print("\nThis script converts M3DRef-CLIP predictions to ScanRefer hidden test benchmark format.")
    main()


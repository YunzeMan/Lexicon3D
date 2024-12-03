"""
This help file adds evaluation labels (easy / hard / view-dep / view-indep) to original Nr3D data, it follows the
official code logic. Please refer to
https://github.com/referit3d/referit3d/blob/eccv/referit3d/analysis/deepnet_predictions.py
"""

from tqdm.contrib.concurrent import process_map
import pandas as pd
import hydra
import ast


def decode_stimulus_string(s):
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = \
            s.split('-', maxsplit=4)
    instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1
    return scene_id, instance_label, n_objects, target_id, distractors_ids


def get_view_dependent_mask(df):
    target_words = {
        'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost', 'looking', 'across'
    }
    return df.tokens.apply(lambda x: len(set(ast.literal_eval(x)).intersection(target_words)) > 0)


def get_easy_mask(df):
    return df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[2]) <= 2


def add_evaluation_labels_to_csv(file_path):
    df = pd.read_csv(file_path)
    easy_mask = get_easy_mask(df)
    view_dependent_mask = get_view_dependent_mask(df)
    df["is_easy"] = easy_mask
    df["is_view_dep"] = view_dependent_mask
    df.to_csv(file_path, index=False)


@hydra.main(version_base=None, config_path="../../config", config_name="global_config")
def main(cfg):
    print("\nDefault: using all CPU cores.")
    file_paths = []
    for split in ("train", "test"):
        file_paths.append(getattr(cfg.data.lang_metadata, f"{split}_language_data"))
    process_map(add_evaluation_labels_to_csv, file_paths, chunksize=1)
    print(f"==> Complete.")


if __name__ == '__main__':
    main()

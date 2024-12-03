import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from m3drefclip.util.utils import get_batch_aabb_pair_ious
from m3drefclip.evaluation.general_evaluator import GeneralEvaluator


class Multi3DReferEvaluator(GeneralEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_types = {"zt_w_d": 0, "zt_wo_d": 1, "st_w_d": 2, "st_wo_d": 3, "mt": 4}

    def _set_ground_truths_from_files(self, path):
        self.ground_truths = {}
        scene_files = os.listdir(path)
        for scene_file in scene_files:
            with open(os.path.join(path, scene_file), "r") as f:
                gt_json = json.load(f)
            for query in gt_json:
                self.ground_truths[(scene_file[:-5], 0, query["ann_id"])] = {
                    "aabb_bound": np.array(query["aabb_bound"], dtype=np.float32),
                    "eval_type": query["eval_type"]
                }

    def evaluate(self, predictions):
        all_gt_info_len = len(self.ground_truths)
        eval_type_mask = np.empty(all_gt_info_len, dtype=np.uint8)
        iou_25_f1_scores = np.zeros(all_gt_info_len, dtype=np.float32)
        iou_50_f1_scores = np.zeros(all_gt_info_len, dtype=np.float32)
        iterator = enumerate(tqdm(predictions.items(), desc="Evaluating") if self.verbose else predictions.items())
        for i, (key, value) in iterator:
            eval_type_mask[i] = self.evaluation_types[self.ground_truths[key]["eval_type"]]
            if self.ground_truths[key]["eval_type"] in ("zt_wo_d", "zt_w_d"):
                iou_25_f1_scores[i] = iou_50_f1_scores[i] = self.evaluate_one_zero_gt_query(value)
            else:
                iou_25_f1_scores[i], iou_50_f1_scores[i] = self.evaluate_one_query(value, self.ground_truths[key])
        iou_25_results = {}
        iou_50_results = {}
        for sub_group in ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt"):
            selected_indices = eval_type_mask == self.evaluation_types[sub_group]
            if np.any(selected_indices):
                # micro-averaged scores of each semantic class and each subtype across queries
                iou_25_results[sub_group] = np.mean(iou_25_f1_scores[selected_indices])
                iou_50_results[sub_group] = np.mean(iou_50_f1_scores[selected_indices])
            else:
                iou_25_results[sub_group] = np.nan
                iou_50_results[sub_group] = np.nan
        iou_25_results["overall"] = np.mean(iou_25_f1_scores)
        iou_50_results["overall"] = np.mean(iou_50_f1_scores)

        if self.verbose:
            self._print_results(iou_25_results, iou_50_results)

        return {f"{self.metric_name}@0.25": iou_25_results, f"{self.metric_name}@0.5": iou_50_results}

    def _print_results(self, iou_25_results, iou_50_results):
        print(f"{'=' * 100}")
        print("{0:<12}{1:<12}{2:<12}{3:<12}{4:<12}{5:<12}{6:<12}"
              .format("IoU", "zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt", "overall"))
        print(f"{'-' * 100}")
        line_1_str = '{:<12}'.format("0.25")
        for sub_group_type, score in iou_25_results.items():
            line_1_str += '{:<12.1f}'.format(score * 100)
        print(line_1_str)
        line_2_str = '{:<12}'.format("0.50")
        for sub_group_type, score in iou_50_results.items():
            line_2_str += '{:<12.1f}'.format(score * 100)
        print(line_2_str)
        print(f"{'=' * 100}\n")

    @staticmethod
    def evaluate_one_query(pred_info, gt_info):
        pred_bboxes_count = len(pred_info["aabb_bound"])
        gt_bboxes_count = len(gt_info["aabb_bound"])

        # initialize true positives
        iou_25_tp = 0
        iou_50_tp = 0

        # initialize the cost matrix
        square_matrix_len = max(gt_bboxes_count, pred_bboxes_count)
        iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float32)
        # TODO: convert to batch process
        for pred_aabb_idx, pred_aabb in enumerate(pred_info["aabb_bound"]):
            for gt_aabb_idx, gt_aabb in enumerate(gt_info["aabb_bound"]):
                iou_matrix[pred_aabb_idx, gt_aabb_idx] = get_batch_aabb_pair_ious(
                    torch.from_numpy(gt_aabb).unsqueeze(0), torch.from_numpy(pred_aabb).unsqueeze(0)
                )

        # apply matching algorithm
        row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)

        # iterate matched pairs, check ious
        for i in range(pred_bboxes_count):
            iou = iou_matrix[row_idx[i], col_idx[i]]
            # calculate true positives
            if iou >= 0.25:
                iou_25_tp += 1
            if iou >= 0.5:
                iou_50_tp += 1

        # calculate precision, recall and f1-score for the current scene
        iou_25_f1_score = 2 * iou_25_tp / (pred_bboxes_count + gt_bboxes_count)
        iou_50_f1_score = 2 * iou_50_tp / (pred_bboxes_count + gt_bboxes_count)
        return iou_25_f1_score, iou_50_f1_score

    @staticmethod
    def evaluate_one_zero_gt_query(pred_info):
        return 1 if len(pred_info["aabb_bound"]) == 0 else 0
import torch


def get_batch_aabb_pair_ious(batch_boxes_1_bound, batch_boxes_2_bound):
    box_1_x_min, box_1_y_min, box_1_z_min = torch.tensor_split(batch_boxes_1_bound[:, 0], 3, dim=1)
    box_1_x_max, box_1_y_max, box_1_z_max = torch.tensor_split(batch_boxes_1_bound[:, 1], 3, dim=1)

    box_2_x_min, box_2_y_min, box_2_z_min = torch.tensor_split(batch_boxes_2_bound[:, 0], 3, dim=1)
    box_2_x_max, box_2_y_max, box_2_z_max = torch.tensor_split(batch_boxes_2_bound[:, 1], 3, dim=1)

    x_a = torch.maximum(box_1_x_min, box_2_x_min)
    y_a = torch.maximum(box_1_y_min, box_2_y_min)
    z_a = torch.maximum(box_1_z_min, box_2_z_min)
    x_b = torch.minimum(box_1_x_max, box_2_x_max)
    y_b = torch.minimum(box_1_y_max, box_2_y_max)
    z_b = torch.minimum(box_1_z_max, box_2_z_max)

    zero_tensor = torch.zeros_like(x_a)
    intersection_volume = torch.maximum((x_b - x_a), zero_tensor) * torch.maximum((y_b - y_a), zero_tensor) * \
                          torch.maximum((z_b - z_a), zero_tensor)
    box_1_volume = (box_1_x_max - box_1_x_min) * (box_1_y_max - box_1_y_min) * (box_1_z_max - box_1_z_min)
    box_2_volume = (box_2_x_max - box_2_x_min) * (box_2_y_max - box_2_y_min) * (box_2_z_max - box_2_z_min)
    iou = intersection_volume / (box_1_volume + box_2_volume - intersection_volume + torch.finfo(torch.float32).eps)
    return iou.flatten()

#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "common_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){

    // Common
    m.def("ballquery_batch_p", &ballquery_batch_p, "ballquery_batch_p");
    m.def("sec_mean", &sec_mean, "sec_mean");
    m.def("sec_min", &sec_min, "sec_min");
    m.def("sec_max", &sec_max, "sec_max");
    m.def("roipool_fp", &roipool_fp, "roipool_fp");
    m.def("roipool_bp", &roipool_bp, "roipool_bp");
    m.def("crop_pcds_from_aabbs", &cropPcdsFromAabbs, "crop_pcds_from_aabbs");
    m.def("get_iou", &get_iou, "get_iou");

    // PointGroup
    m.def("pg_bfs_cluster", &pg_bfs_cluster, "pg_bfs_cluster");

}
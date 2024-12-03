/*
Ball Query with BatchIdx & Clustering Algorithm
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "bfs_cluster.h"

/* ================================== ballquery_batch_p ================================== */
// input xyz: (n, 3) float
// input batch_idxs: (n) int
// input batch_offsets: (B+1) int, batch_offsets[-1]
// output idx: (n * meanActive) dim 0 for number of points in the ball, idx in n
// output start_len: (n, 2), int
int ballquery_batch_p(at::Tensor xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor batch_offsets_tensor, at::Tensor idx_tensor, at::Tensor start_len_tensor, int n, int meanActive, float radius){
    const float *xyz = xyz_tensor.data_ptr<float>();
    const uint8_t *batch_idxs = batch_idxs_tensor.data_ptr<uint8_t>();
    const int *batch_offsets = batch_offsets_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    int *start_len = start_len_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int cumsum = ballquery_batch_p_cuda(n, meanActive, radius, xyz, batch_idxs, batch_offsets, idx, start_len, stream);
    return cumsum;
}

/* ================================== bfs_cluster ================================== */
ConnectedComponent pg_find_cc(Int idx, int16_t *semantic_label, Int *ball_query_idxs, int *start_len, int *visited){
    ConnectedComponent cc;
    cc.addPoint(idx);
    visited[idx] = 1;

    std::queue<Int> Q;
    assert(Q.empty());
    Q.push(idx);

    while(!Q.empty()){
        Int cur = Q.front(); Q.pop();
        int start = start_len[cur * 2];
        int len = start_len[cur * 2 + 1];
        int16_t label_cur = semantic_label[cur];
        for(Int i = start; i < start + len; i++){
            Int idx_i = ball_query_idxs[i];
            if(semantic_label[idx_i] != label_cur) continue;
            if(visited[idx_i] == 1) continue;

            cc.addPoint(idx_i);
            visited[idx_i] = 1;

            Q.push(idx_i);
        }
    }
    return cc;
}

//input: semantic_label, int16, N
//input: ball_query_idxs, Int, (nActive)
//input: start_len, int, (N, 2)
//output: clusters, CCs
int pg_get_clusters(int16_t *semantic_label, Int *ball_query_idxs, int *start_len, const Int nPoint, int threshold, ConnectedComponents &clusters){
    int visited[nPoint] = {0};

    int sumNPoint = 0;
    for(Int i = 0; i < nPoint; i++){
        if(visited[i] == 0){
            ConnectedComponent CC = pg_find_cc(i, semantic_label, ball_query_idxs, start_len, visited);
            if((int)CC.pt_idxs.size() >= threshold){
                clusters.push_back(CC);
                sumNPoint += (int)CC.pt_idxs.size();
            }
        }
    }

    return sumNPoint;
}

void fill_cluster_idxs_(ConnectedComponents &CCs, int *cluster_obj_idxs, long *cluster_point_idxs, int *cluster_offsets){
    for(int i = 0; i < (int)CCs.size(); i++) {
        cluster_offsets[i + 1] = cluster_offsets[i] + (int)CCs[i].pt_idxs.size();
        for(int j = 0; j < (int)CCs[i].pt_idxs.size(); j++) {
            long idx = (long)CCs[i].pt_idxs[j];
            int tmp_idx = cluster_offsets[i] + j;
            cluster_obj_idxs[tmp_idx] = i;
            cluster_point_idxs[tmp_idx] = idx;
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> pg_bfs_cluster(at::Tensor semantic_label_tensor, at::Tensor ball_query_idxs_tensor, at::Tensor start_len_tensor, const int N, int threshold){
    int16_t *semantic_label = semantic_label_tensor.data_ptr<int16_t>();
    Int *ball_query_idxs = ball_query_idxs_tensor.data_ptr<Int>();
    int *start_len = start_len_tensor.data_ptr<int>();
    ConnectedComponents CCs;
    int sumNPoint = pg_get_clusters(semantic_label, ball_query_idxs, start_len, N, threshold, CCs);
    int nCluster = (int)CCs.size();

    at::Tensor cluster_obj_idxs_tensor = torch::zeros({sumNPoint}, torch::kInt32);
    at::Tensor cluster_point_idxs_tensor = torch::zeros({sumNPoint}, torch::kInt64);
    at::Tensor cluster_offsets_tensor = torch::zeros({nCluster + 1}, torch::kInt32);

//    cluster_idxs_tensor.resize_({sumNPoint, 2});

    // cluster_offsets_tensor.resize_({nCluster + 1});

    int *cluster_obj_idxs = cluster_obj_idxs_tensor.data_ptr<int>();
    long *cluster_point_idxs = cluster_point_idxs_tensor.data_ptr<long>();
    int *cluster_offsets = cluster_offsets_tensor.data_ptr<int>();

    fill_cluster_idxs_(CCs, cluster_obj_idxs, cluster_point_idxs, cluster_offsets);

    return std::make_tuple(cluster_obj_idxs_tensor, cluster_point_idxs_tensor, cluster_offsets_tensor);
}
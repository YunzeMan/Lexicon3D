#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    ## <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      ## <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbsh-delta-gpu
#SBATCH --job-name=train01-sd21-l1
#SBATCH --time=48:00:00      ## hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=yunzem2@illinois.edu
#SBATCH -o /scratch/bbsh/yunzem2/logs/3dllm/lexicon3d/feature-extraction-fusion-sd21-l1-train01.out


. ~/.bashrc

module reset 
module list  # job documentation and metadata
module unload anaconda3_gpu
echo "job is starting on `hostname`"

conda activate /scratch/bbsg/yunzem2/conda-envs/videodiff
cd /u/yunzem2/codes/lexicon3d/lseg_feature_extraction/scripts/

# python fusion_scannet_swin3d.py   --output_dir  /scratch/bbsg/yunzem2/dataset/lexicon3d/swin3d/  --process_id_range 0,2 --voxel_size 0.05 --prefix swin3d
python fusion_scannet_swin3d.py   --output_dir  /scratch/bbsg/yunzem2/dataset/lexicon3d/swin3d-pcd1/  --process_id_range 0,2 --voxel_size 0.05 --prefix swin3d-pcd1

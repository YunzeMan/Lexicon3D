#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    ## <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      ## <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbsg-delta-gpu
#SBATCH --job-name=clip-m-40-train01
#SBATCH --time=48:00:00      ## hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=yunzem2@illinois.edu
#SBATCH -o /scratch/bbsh/yunzem2/logs/3dllm/lexicon3d/feature-extraction-fusion-clip-m-40-train01.out


. ~/.bashrc

module reset 
module list  # job documentation and metadata
module unload anaconda3_gpu
echo "job is starting on `hostname`"

conda activate /scratch/bbsg/yunzem2/conda-envs/videodiff
cd /u/yunzem2/codes/lexicon3d/lseg_feature_extraction/scripts/

python fusion_scannet_clip.py  --data_dir /scratch/bbsg/yunzem2/dataset/ScanNet/openscene/  --output_dir  /scratch/bbsg/yunzem2/dataset/lexicon3d/clip-multiscale/  --process_id_range 600,1299  --split train  --voxel_size 0.05 --prefix clip-multiscale

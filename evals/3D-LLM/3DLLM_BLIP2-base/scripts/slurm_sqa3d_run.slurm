#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    ## <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      ## <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=YOUR_JOB_NAME
#SBATCH --time=48:00:00      ## hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH -o YOUR_JOB_NAME.out


. ~/.bashrc

module reset 
module list  # job documentation and metadata
module unload anaconda3_gpu
echo "job is starting on `hostname`"

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/finetune_sqa.yaml

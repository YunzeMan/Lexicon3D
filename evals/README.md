# Evaluation on Downstream Tasks 
---
This folder contains the evaluation scripts for the downstream tasks. For each task, we provide minimal codes required to evaluate the extracted features. For more details, please refer to the original repositories of the downstream tasks. 

### 3D Scene Reasoning: Question Answering

For the scene reasoning task, we use the 3D-LLM model to evaluate the extracted features. The 3D-LLM model is a large transformer-based model that takes the scene features as input and predicts the answers to the questions.

**For Installation and Requirements**, please refer to the [3D-LLM repository](https://github.com/UMass-Foundation-Model/3D-LLM).

**For Evaluation**, we provide the slurm scripts for evaluation at `evals/3D-LLM/3DLLM_BLIP2-base/scripts/slurm_sqa3d_run.slurm`. You can run the evaluation script with the following command:

```bash
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/finetune_sqa.yaml
```

The core modification is to replace the input features with the extracted features from the foundation models. To achieve this, you need to modify the configuration files in `lavis/projects/blip2/train` to specify the location to the extracted features and the visual feature channels.

In addition, the dataloaders have to be slightly modified to load the extracted features. You can refer to the `lavis/datasets/datasets/threedvqa_datasets.py` for the modified dataloader implementation.
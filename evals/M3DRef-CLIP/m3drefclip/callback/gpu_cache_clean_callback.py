from lightning.pytorch.callbacks import Callback
import torch


class GPUCacheCleanCallback(Callback):
    def on_train_batch_start(self, *args, **kwargs):
        torch.cuda.empty_cache()

    def on_validation_batch_start(self, *args, **kwargs):
        torch.cuda.empty_cache()

    def on_test_batch_start(self, *args, **kwargs):
        torch.cuda.empty_cache()

    def on_predict_batch_start(self, *args, **kwargs):
        torch.cuda.empty_cache()

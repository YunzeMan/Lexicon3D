from lightning.pytorch.callbacks import Callback
from math import cos, pi


class LrDecayCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # cosine learning rate decay
        current_epoch = trainer.current_epoch
        start_epoch = pl_module.hparams.cfg.model.lr_decay.start_epoch
        if trainer.current_epoch < start_epoch:
            return
        end_epoch = pl_module.hparams.cfg.trainer.max_epochs
        clip = 1e-6
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = clip + 0.5 * (pl_module.hparams.cfg.model.optimizer.lr - clip) * \
                                (1 + cos(pi * ((current_epoch - start_epoch) / (end_epoch - start_epoch))))

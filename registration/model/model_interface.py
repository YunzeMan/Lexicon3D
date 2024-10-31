# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import inspect
import torch
import numpy as np
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
from .metrics import SemsegMeter
from typing import List, Union
from torch import Tensor

_EPS = 1e-6


def se3_init(rot, trans):
    pose = np.concatenate([rot, trans], axis=-1)
    return pose

def se3_inv(pose):
    """Inverts the SE3 transform"""
    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    irot = rot.transpose(-1, -2)
    itrans = -irot @ trans
    return se3_init(irot, itrans)

def se3_cat(a, b):
    """Concatenates two SE3 transforms"""
    #print(a, b)
    rot_a, trans_a = a[..., :3, :3], a[..., :3, 3:4]
    rot_b, trans_b = b[..., :3, :3], b[..., :3, 3:4]

    rot = rot_a @ rot_b
    trans = rot_a @ trans_b + trans_a
    dst = se3_init(rot, trans)
    return dst


def se3_compare(a, b):
    combined = torch.from_numpy(se3_cat(a, se3_inv(b)))

    trace = combined[..., 0, 0] + combined[..., 1, 1] + combined[..., 2, 2]
    rot_err_deg = torch.acos(torch.clamp(0.5 * (trace - 1), -1., 1.)) \
                  * 180 / math.pi
    trans_err = torch.norm(combined[..., :, 3], dim=-1)

    err = {
        'rot_deg': rot_err_deg,
        'trans': trans_err
    }
    return err


def se3_transform_list(pose: Union[List[Tensor], Tensor], xyz: List[Tensor]):
    """Similar to se3_transform, but processes lists of tensors instead

    Args:
        pose: List of (3, 4)
        xyz: List of (N, 3)

    Returns:
        List of transformed xyz
    """

    B = len(xyz)
    assert all([xyz[b].shape[-1] == 3 and pose[b].shape[:-2] == xyz[b].shape[:-2] for b in range(B)])

    transformed_all = []
    for b in range(B):
        rot, trans = pose[b][..., :3, :3].cuda(), pose[b][..., :3, 3:4].cuda()
        transformed = torch.einsum('...ij,...bj->...bi', rot, xyz[b]) + trans.transpose(-1, -2)  # Rx + t
        transformed_all.append(transformed)

    return transformed_all


class CorrCriterion(nn.Module):
    """Correspondence Loss.
    """
    def __init__(self, metric='mae'):
        super().__init__()
        assert metric in ['mse', 'mae']

        self.metric = metric

    def forward(self, kp_before, kp_warped_pred, pose_gt, overlap_weights=None):

        losses = {}
        B = pose_gt.shape[0]

        kp_warped_gt = se3_transform_list(pose_gt, kp_before)
        corr_err = torch.cat(kp_warped_pred, dim=0) - torch.cat(kp_warped_gt, dim=0)

        if self.metric == 'mae':
            corr_err = torch.sum(torch.abs(corr_err), dim=-1)
        elif self.metric == 'mse':
            corr_err = torch.sum(torch.square(corr_err), dim=-1)
        else:
            raise NotImplementedError

        #print(corr_err.shape, overlap_weights.shape)
        #overlap_weights = torch.squeeze(overlap_weights, 2)
        if overlap_weights is not None:
            #overlap_weights = torch.cat(overlap_weights)
            mean_err = torch.sum(overlap_weights * corr_err) / torch.clamp_min(torch.sum(overlap_weights), _EPS)
        else:
            mean_err = torch.mean(corr_err, dim=1)

        return mean_err


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.overlap_criterion = nn.BCEWithLogitsLoss()
        self.corr_criterion = CorrCriterion(metric='mae')

        # # Project-Specific Definitions
        # self.hsi_index = np.r_[0, 4:12]
        # self.rgb_index = (3, 2, 1)
        self.metrics = SemsegMeter()

    def _compute_metrics(self, pred, gt):

        metrics = {}
        with torch.no_grad():
            pose_err = se3_compare(pred, gt)
            metrics['rot_err_deg'] = pose_err['rot_deg']
            metrics['trans_err'] = pose_err['trans']

        return metrics
    

    def forward(self, feat_1, pos_1, feat_2, pos_2):
        return self.model(feat_1, pos_1, feat_2, pos_2)

    def training_step(self, batch, batch_idx):
        feat_1, pos_1, feat_2, pos_2, overlap_gt, pose_gt = batch
        src_overlap_gt, tgt_overlap_gt = torch.split(overlap_gt, [feat_1.shape[1], feat_2.shape[1]], 1)
        
        pred = self(feat_1, pos_1, feat_2, pos_2)

        pred_src_overlap = pred['src_overlap'][0]
        pred_tgt_overlap = pred['tgt_overlap'][0]

        pred_merged = torch.squeeze(torch.cat((pred_src_overlap, pred_tgt_overlap), 1), 2)
        #print(pred_merged, overlap_gt)
        loss_overlap = self.overlap_criterion(pred_merged, overlap_gt.float())


        src_corr_loss = self.corr_criterion(
            pred['src_kp'],
            pred['src_kp_warped'],
            pose_gt,
            overlap_weights=src_overlap_gt
        )
        tgt_corr_loss = self.corr_criterion(
            pred['tgt_kp'],
            pred['tgt_kp_warped'],
            torch.stack([torch.from_numpy(se3_inv(p.cpu())) for p in pose_gt]),
            overlap_weights=tgt_overlap_gt
        )
        

        loss_corr = src_corr_loss + tgt_corr_loss

        #label = torch.nn.functional.one_hot(label[0], num_classes=20)
        #print(pred.shape, label.max())
        #loss = self.loss_function(pred[0], label[0].type(torch.LongTensor).cuda())
        loss = loss_overlap + loss_corr
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_corr', loss_corr, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_overlap', loss_overlap, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_start(self):
        self.rot_errors = []
        self.trans_errors = []

    def validation_step(self, batch, batch_idx):
        feat_1, pos_1, feat_2, pos_2, overlap_gt, pose_gt = batch
        src_overlap_gt, tgt_overlap_gt = torch.split(overlap_gt, [feat_1.shape[1], feat_2.shape[1]], 1)
        
        pred = self(feat_1, pos_1, feat_2, pos_2)

        #pred = self.forward(batch)
        #losses = self.compute_loss(pred, batch)
        #print(pred, pose_gt)
        metrics = self._compute_metrics(pred['pose'].cpu(), pose_gt.cpu())

        # visualize_registration(batch, pred, metrics=metrics, iter_idx=5, b=2)

        val_outputs = metrics
        #print(metrics['rot_err_deg'].shape)
        self.rot_errors.append(metrics['rot_err_deg'][0][0])
        self.trans_errors.append(metrics['trans_err'][0][0])

        return val_outputs
    
        #pass
        # feat_1, pos_1, feat_2, pos_2, overlap = batch

        # pred = self(feat)
        # pred_idx = torch.argmax(pred, -1)
        # self.metrics.update(pred_idx[0], label[0])


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        final_metrics = {}
        #print(self.rot_errors, self.trans_errors)
        final_metrics['rot_err'] = np.mean(np.array(self.rot_errors))
        final_metrics['trans_err'] = np.mean(np.array(self.trans_errors))
        with open(self.hparams.model_name + "_logeval.txt", 'a+') as f:
            f.write(str(final_metrics['rot_err']) + " " + str(final_metrics['trans_err']))
            f.write("\n")
        print("metrics:", final_metrics)
        # self.print(self.get_progress_bar_dict())

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        for name, params in self.named_parameters():
            print(name, params)
        print(self.hparams.lr_scheduler, "scheduler!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'crossentropy':
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

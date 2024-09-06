from diffusers import StableVideoDiffusionPipeline
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models import UNetSpatioTemporalConditionModel
from diffusers import DDIMScheduler, EulerDiscreteScheduler
import gc
import os
from PIL import Image
from torchvision.transforms import PILToTensor

import torch.nn.functional as F

from einops import rearrange
import math
import random

class MyUNetSpatioTemporalConditionModel(UNetSpatioTemporalConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor):
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead of a plain
                tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb
        # emb = torch.zeros_like(emb)
        # emb = emb.zeros_like()

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # emb = torch.zeros_like(emb)

        # emb = None
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )
            if i in up_ft_indices:
                up_ft[i] = sample.detach()

        output = {}
        output["up_ft"] = up_ft
        return output

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)

def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out



class OneStepSVDPipeline(StableVideoDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        ref_img_tensor,
        fps = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
    ):

        device = self._execution_device

        ref_img_resized = _resize_with_antialiasing(ref_img_tensor,(224,224))
        ref_img_resized = ref_img_resized / 2 + 0.5

        image_embeddings = self._encode_image(ref_img_resized, device, 1, False)

        batch_size = img_tensor.shape[0]

        fps = fps - 1

        added_time_ids = self._get_add_time_ids(
            fps,
            127,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            1,
            False,
        )
        added_time_ids = added_time_ids.to(device)

        # ?
        b, f, c, h, w = img_tensor.shape
        img_tensor = rearrange(img_tensor, 'b f c h w -> (b f) c h w')
        

        latents = self.vae.encode(img_tensor).latent_dist.mode() * self.vae.config.scaling_factor

        latents = rearrange(latents, '(b f) c h w -> b f c h w', b=b, f=f)
        noise = torch.randn_like(latents).to(device)

        timesteps = self.scheduler.timesteps
        timestep = timesteps[-1-t].view(1,).to(device)

        latents_noisy = self.scheduler.add_noise(latents, noise, timestep)
        latents_noisy = self.scheduler.scale_model_input(latents_noisy, timestep)
 
        noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))

        ref_img_tensor = ref_img_tensor + noise_aug_strength * torch.randn_like(ref_img_tensor)
        # import pdb; pdb.set_trace()
        conditional_latents = self.vae.encode(ref_img_tensor).latent_dist.mode()


        conditional_latents = conditional_latents.unsqueeze(1).repeat(1, latents_noisy.shape[1], 1, 1, 1)
        unconditional_latents = torch.zeros_like(conditional_latents)
        # con_input = torch.cat([latents_noisy, latents/self.vae.config.scaling_factor], dim=2)
        uncon_input = torch.cat([latents_noisy, unconditional_latents], dim=2)

        unet_output = self.unet(
            sample = uncon_input,
            timestep = timestep,
            up_ft_indices = up_ft_indices,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
        )
        # unet_output = self.unet(
        #     sample = con_input,
        #     timestep = timestep,
        #     up_ft_indices = up_ft_indices,
        #     encoder_hidden_states=image_embeddings,
        #     added_time_ids=added_time_ids,
        # )
        return unet_output


class SVDFeaturizer:
    # def __init__(self, sd_id='stabilityai/stable-video-diffusion-img2vid-xt'):
    def __init__(self, sd_id='stabilityai/stable-video-diffusion-img2vid-xt-1-1'):
        unet = MyUNetSpatioTemporalConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSVDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = EulerDiscreteScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self,
                img_tensor,
                ref_img_tensor,
                t=261,
                up_ft_index=2,
                fps = 7,
                motion_bucket_id: int = 127,
                noise_aug_strength: float = 0.02,
                ):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''
        # img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        # img_tensor_shape: bs, nf, c, h, w
        img_tensor = img_tensor.cuda()

        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            ref_img_tensor = ref_img_tensor)

        unet_ft = unet_ft_all['up_ft'][up_ft_index] # b, c, h, w
        return unet_ft
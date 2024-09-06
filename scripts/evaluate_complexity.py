import torch
from torch import nn
import time
import sys
sys.path.append('..')
# from additional_utils.models import LSeg_MultiEvalModule
# from modules.lseg_module import LSegModule
# from encoding.models.sseg import BaseNet
# from video_diffusion.dift_sd import SDFeaturizer
# import open_clip
# import einops as E
# from utils import center_padding, resize_pos_embed, tokens_to_output
# from video_diffusion.dift_svd import SVDFeaturizer


class CLIP(nn.Module):
    def __init__(
        self,
        arch="ViT-L-14",
        checkpoint="openai",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()
        assert output in ["dense-cls", "cls", "gap", "dense"]
        self.output = output
        self.checkpoint_name = "clip_" + arch.replace("-", "").lower() + checkpoint

        # Initialize a pre-trained CLIP image encoder and freeze it.
        _clip_model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=checkpoint
        )
        _clip_model = _clip_model.eval().to(torch.float32)
        self.visual = _clip_model.visual
        del _clip_model

        # Extract some attributes from CLIP module for easy access.
        self.patch_size = self.visual.conv1.stride[0]

        # get feature dimension
        feat_dim = self.visual.transformer.width
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim
        feat_dims = [feat_dim, feat_dim, feat_dim, feat_dim]

        # get extraction targets
        n_layers = len(self.visual.transformer.resblocks)
        multilayers = [
            n_layers // 4 - 1,
            n_layers // 2 - 1,
            n_layers // 4 * 3 - 1,
            n_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dims
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # clip stuff
        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")

        # concat cls token
        _cls_embed = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([_cls_embed.to(x.dtype), x], dim=1)

        # add pos embed
        pos_embed = resize_pos_embed(self.visual.positional_embedding, x_hw)
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))

        embeds = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, _x in enumerate(embeds):
            _x = tokens_to_output(self.output, _x[:, 1:], _x[:, 0], out_hw)
            outputs.append(_x)

        return outputs[0] if len(outputs) == 1 else outputs


def run_model(model, model_type, input_tensor):
    if model_type == "dinov2":
        return model.forward_features(input_tensor)
    elif model_type == "lseg":
        return model.parallel_forward(input_tensor, '')
    elif model_type == "sd":
        return model.forward(input_tensor, t=100, up_ft_index=1)
    elif model_type == "clip":
        return model(input_tensor)
    elif model_type == "vjepa":
        return model(input_tensor)
    elif model_type == "svd":
        return model.forward(input_tensor, input_tensor[:, 0], t=25, up_ft_index=1)
    else:
        raise ValueError("Invalid model type")

# Function to measure the memory usage
def measure_memory(model, model_type, input_tensor):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = run_model(model, model_type, input_tensor)
    memory_allocated = torch.cuda.max_memory_allocated()
    return memory_allocated / (1024 ** 2)  # Convert to MB

# Function to measure the inference time
def measure_inference_time(model, model_type, input_tensor, num_steps=1000):
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = run_model(model, model_type, input_tensor)
        
        # Measure time
        start_time = time.time()
        for _ in range(num_steps):
            _ = run_model(model, model_type, input_tensor)
        end_time = time.time()
    
    avg_time_per_step = (end_time - start_time) / num_steps
    return avg_time_per_step


if __name__ == "__main__":
    # model_type = "dinov2"  # Change this to the type of your model
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda() # 238, 322 --> 17, 23
    # dummy_input = torch.randn(1, 3, 238, 322).cuda()  # Example shape, change it as needed
    # model.eval()

    # model_type = "lseg"
    # module = LSegModule.load_from_checkpoint(
    #     checkpoint_path='/scratch/bbsg/yunzem2/dataset/lexicon3d/lseg/lseg_checkpoint/demo_e200.ckpt',
    #     data_path='../datasets/',
    #     dataset='ade20k',
    #     backbone='clip_vitl16_384',
    #     aux=False,
    #     num_features=256,
    #     aux_weight=0,
    #     se_loss=False,
    #     se_weight=0,
    #     base_lr=0,
    #     batch_size=1,
    #     max_epochs=0,
    #     ignore_index=255,
    #     dropout=0.0,
    #     scale_inv=False,
    #     augment=False,
    #     no_batchnorm=False,
    #     widehead=True,
    #     widehead_hr=False,
    #     map_locatin="cpu",
    #     arch_option=0,
    #     block_depth=0,
    #     activation='lrelu',)
    # if isinstance(module.net, BaseNet):
    #     model = module.net
    # else:
    #     model = module
    # scales = ([1])
    # model.crop_size = 640
    # model.base_size = 640
    # model = LSeg_MultiEvalModule(model, scales=scales, flip=True).cuda() # LSeg model has to be in GPU
    # model = model.eval()
    # dummy_input = torch.randn(1, 3, 240, 320).cuda()  # Example shape, change it as needed
    
    # model_type = "clip"
    # model = CLIP(arch="ViT-L-14", checkpoint="openai", output="dense", layer=-1, return_multilayer=False).cuda() # 238, 322 --> 17, 23
    # dummy_input = torch.randn(1, 3, 238, 322).cuda()  # Example shape, change it as needed

    # model_type = "sd"
    # model = SDFeaturizer()
    # dummy_input = torch.randn(1, 1, 3, 240, 320)  # Example shape, change it as needed

    # model_type = "vjepa"
    # model = build_jepa()
    # model = model.cuda()
    # dummy_input = torch.randn(1, 3, 16, 240, 320).cuda()  # Example shape, change it as needed

    # model_type = "svd"
    # model = SVDFeaturizer()
    # dummy_input = torch.randn(1, 16, 3, 256, 320).cuda()  # Example shape, change it as needed


    # Measure memory usage
    memory_usage_mb = measure_memory(model, model_type, dummy_input)
    print(f"Memory Usage: {memory_usage_mb:.2f} MB")

    # Measure inference time
    avg_inference_time = measure_inference_time(model, model_type, dummy_input, num_steps=100)
    print(f"Average Inference Time: {avg_inference_time:.6f} seconds per step")

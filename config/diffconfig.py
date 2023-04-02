from typing import Optional, List
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool
from tensorfn.config import (
    MainConfig,
    Config,
    Optimizer,
    Scheduler,
    DataLoader,
    Instance,
)

import diffusion
import model
from models.unet_autoenc import BeatGANsAutoencConfig


class Diffusion(Config):
    beta_schedule: Instance

class Dataset(Config):
    name: StrictStr
    path: StrictStr
    resolution: StrictInt

class Training(Config):
    ckpt_path: StrictStr
    optimizer: Optimizer
    scheduler: Optional[Scheduler]
    dataloader: DataLoader


class Eval(Config):
    wandb: StrictBool
    save_every: StrictInt
    valid_every: StrictInt
    log_every: StrictInt


class DiffusionConfig(MainConfig):
    diffusion: Diffusion
    training: Training


def get_model_conf():

    return BeatGANsAutoencConfig(image_size=256, 
    in_channels=3+20, 
    model_channels=128, 
    out_channels=3*2,  # also learns sigma
    num_res_blocks=2, 
    num_input_res_blocks=None, 
    embed_channels=512, 
    attention_resolutions=(32, 16, 8,), 
    time_embed_channels=None, 
    dropout=0.1, 
    channel_mult=(1, 1, 2, 2, 4, 4), 
    input_channel_mult=None, 
    conv_resample=True, 
    dims=2, 
    num_classes=None, 
    use_checkpoint=False,
    num_heads=1, 
    num_head_channels=-1, 
    num_heads_upsample=-1, 
    resblock_updown=True, 
    use_new_attention_order=False, 
    resnet_two_cond=True, 
    resnet_cond_channels=None, 
    resnet_use_zero_module=True, 
    attn_checkpoint=False, 
    enc_out_channels=512, 
    enc_attn_resolutions=None, 
    enc_pool='adaptivenonzero', 
    enc_num_res_block=2, 
    enc_channel_mult=(1, 1, 2, 2, 4, 4, 4), 
    enc_grad_checkpoint=False, 
    latent_net_conf=None)
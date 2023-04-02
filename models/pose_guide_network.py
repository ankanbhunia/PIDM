from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu
from torch import nn

from .unet import *
from choices import *
from .blocks import *
from .latentnet import *
from einops import rearrange, reduce, repeat


@dataclass
class BeatGANsPoseGuideConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsPoseGuideModel(self)


class BeatGANsPoseGuideModel(nn.Module):

    def __init__(self, conf: BeatGANsPoseGuideConfig):
        super().__init__()
        self.conf = conf

        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )
        
        self.ref_encoder = BeatGANsEncoder(conf)
        self.xt_encoder = BeatGANsEncoder(conf)
        
        conf.in_channels = 20
        self.pose_encoder = BeatGANsEncoder(conf)

        self.cros_attn1 = AttentionBlock(channels = 512)
        self.cros_attn2 = AttentionBlock(channels = 512)

        self.self_attn = AttentionBlock_self(channels = 512)
        self.token = nn.Parameter(torch.randn((512)))

        self.linear = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 4),
            )

    def forward(self,
                xt,
                ref,
                pose,
                t,
               ):

        emb_t = self.time_embed(timestep_embedding(t,  self.conf.model_channels))
        ref_feats = self.ref_encoder(ref, t = emb_t)
        pose_feats = self.pose_encoder(pose, t = emb_t)
        xt_feats = self.xt_encoder(xt, t = emb_t)

        ref_out = self.cros_attn1(x = xt_feats[-1], cond = ref_feats[-1]).mean([2,3])

        pose_out = self.cros_attn2(x = xt_feats[-1], cond = pose_feats[-1]).mean([2,3])

        logits = self.linear(torch.cat([ref_out,pose_out],1))


        # pose_out = rearrange(pose_out, 'b c h w -> b c (h w)')
        # ref_out = rearrange(ref_out, 'b c h w -> b c (h w)')
        
        # tkn = self.token.repeat(ref.shape[0],1).unsqueeze(-1)

        # concat_out = torch.cat([tkn, pose_out, ref_out], 2)
        
        # out = self.self_attn(concat_out)[:,:,0]

        # logits = self.linear(out)

        return logits


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)

        return time_emb
